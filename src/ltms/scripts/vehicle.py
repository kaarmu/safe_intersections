#! /usr/bin/env python3

import numpy as np
import random
from time import time
from collections import deque
from datetime import datetime, timedelta
from threading import Thread
from queue import SimpleQueue

# SVEA imports
from svea.states import VehicleState
from svea.interfaces import LocalizationInterface, ActuationInterface, PlannerInterface
from svea.controllers.pure_pursuit import PurePursuitController
from svea.data import RVIZPathHandler
from svea.simulators.sim_SVEA import SimSVEA
from svea.models.bicycle import SimpleBicycleModel
from svea_msgs.msg import lli_ctrl
from svea_msgs.msg import VehicleState as VehicleStateMsg
from svea_mocap.mocap import MotionCaptureInterface
from svea_planners.astar import AStarPlanner, AStarWorld

# ROS imports
import rospy
from ltms.srv import Connect, Notify, Reserve
import message_filters as mf
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, TwistStamped, PointStamped
from nav_msgs.msg import Path
from tf.transformations import quaternion_from_euler, euler_from_quaternion

import tf2_ros
import tf2_geometry_msgs
from tf import transformations 

def state_to_pose(state):
    pose = PoseStamped()
    pose.header = state.header
    pose.pose.position.x = state.x
    pose.pose.position.y = state.y
    qx, qy, qz, qw = quaternion_from_euler(0, 0, state.yaw)
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw
    return pose

def pose_to_state(pose):
    state = VehicleStateMsg()
    state.header = pose.header
    state.x = pose.pose.position.x
    state.y = pose.pose.position.y
    roll, pitch, yaw = euler_from_quaternion([pose.pose.orientation.x,
                                              pose.pose.orientation.y,
                                              pose.pose.orientation.z,
                                              pose.pose.orientation.w])
    state.yaw = yaw
    return state

def load_param(name, value=None):
    """Function used to get parameters from ROS parameter server

    :param name: name of the parameter
    :type name: string
    :param value: default value of the parameter, defaults to None
    :type value: _type_, optional
    :return: value of the parameter
    :rtype: _type_
    """
    if value is None:
        assert rospy.has_param(name), f'Missing parameter "{name}"'
    return rospy.get_param(name, value)

class Vehicle:
    
    NUM_SESSIONS = 5

    DELTA_TIME = 0.1
    LOOP_TIME = 3.0
    
    MIN_VEL = 0.4
    MAX_VEL = 0.8
    MIN_ACC = -0.2
    MAX_ACC = 0.2
    MIN_STEER = -np.pi/4
    MAX_STEER = np.pi/4
    MIN_STEER_RATE = -np.pi/4
    MAX_STEER_RATE = np.pi/4
    
    STATE_DIMS = 4 # or 5 if including steering rate
    if STATE_DIMS == 4:
        ACC_GRID, STEER_GRID = np.meshgrid(np.linspace(MIN_ACC, MAX_ACC), np.linspace(MIN_STEER, MAX_STEER))
    elif STATE_DIMS == 5:
        ACC_GRID, STEER_RATE_GRID = np.meshgrid(np.linspace(MIN_ACC, MAX_ACC), np.linspace(MIN_STEER_RATE, MAX_STEER_RATE))
    
    NUM_ENTRIES = 8
    NUM_EXITS = 4
    ENTRY_HEADINGS = np.linspace(0, 2*np.pi, NUM_ENTRIES, endpoint=False)
    EXIT_HEADINGS = np.linspace(0, 2*np.pi, NUM_EXITS, endpoint=False)
    MAX_ENTRY_DELTA = np.pi/4
    ENTRY_TIMES = np.arange(0, NUM_SESSIONS*LOOP_TIME, LOOP_TIME)
    EXIT_TIMES = np.arange(LOOP_TIME, NUM_SESSIONS*LOOP_TIME+LOOP_TIME, LOOP_TIME)
    
    LIMITS_SHAPE = (50, 50)

    DEBUG = True

    def __init__(self):

        ## Initialize node

        rospy.init_node(self.__class__.__name__)

        ## Load parameters

        self.NAME = load_param('~name', 'svea')
        self.AREA = load_param('~area', 'sml')
        self.INIT_STATE = load_param("~init_state", [4, 4, np.pi*3/4, 0])
        self.INIT_WAIT = load_param('~init_wait', 5)

        # initial state
        self.steer = 0.0
        self.state = VehicleState(*self.INIT_STATE)

        # rate
        self.rate = rospy.Rate(10)

        # initialize motion capture
        self.init_mocap()

        ## Create service proxies
        self.connect_srv = rospy.ServiceProxy('connect', Connect)

        ## Node initialized

        rospy.loginfo(f'{self.__class__.__name__} initialized!')
        
    def init_session(self, id):
        def limits_cb(msg):
            limits = np.frombuffer(msg.data, dtype=np.float32)
            self.sessions[id]['limits'] = limits.reshape(self.LIMITS_SHAPE)
            
        name = f'{self.NAME}_{id}'
        entry_time = self.init_time + rospy.Duration(self.ENTRY_TIMES[id])
        exit_time = self.init_time + rospy.Duration(self.EXIT_TIMES[id])
        ## Connect to LTMS
        resp = self.connect_srv(name, entry_time)
        if not resp.success:
            rospy.logerr(f'Failed to connect to LTMS: {resp.message}')
            return
        ## Set up session
        self.sessions[id]['name'] = name
        self.sessions[id]['entry_time'] = entry_time
        self.sessions[id]['exit_time'] = exit_time
        self.sessions[id]['its_id'] = resp.its_id
        self.sessions[id]['states_pub'] = rospy.Publisher(f'{name}/states', VehicleStateMsg, queue_size=1)
        self.sessions[id]['notify_srv'] = rospy.ServiceProxy(f'{resp.its_id}/notify', Notify)
        self.sessions[id]['reserve_srv'] = rospy.ServiceProxy(f'{resp.its_id}/reserve', Reserve)
        self.sessions[id]['limits_sub'] = rospy.Subscriber(f'{resp.its_id}/limits', bytes, limits_cb)
        self.sessions[id]['valid'] = True
    
    def control_update(self, limits_mask, steer_rate=False):
        if not steer_rate:
            steer_range = (self.steer + self.MIN_STEER_RATE*self.DELTA_TIME, self.steer + self.MAX_STEER_RATE*self.DELTA_TIME)
            valid_limits = limits_mask & (self.ACC_GRID >= self.MIN_ACC) & (self.ACC_GRID <= self.MAX_ACC) & (self.STEER_GRID >= steer_range[0]) & (self.STEER_GRID <= steer_range[1])
        else:
            valid_limits = limits_mask & (self.ACC_GRID >= self.MIN_ACC) & (self.ACC_GRID <= self.MAX_ACC) & (self.STEER_RATE_GRID >= self.MIN_STEER_RATE) & (self.STEER_RATE_GRID <= self.MAX_STEER_RATE)
        # select a random control from the valid limits
        i, j = np.random.choice(np.argwhere(valid_limits))
        vel = self.state.v + self.ACC_GRID[i, j]*self.DELTA_TIME
        steer = self.steer + self.STEER_RATE_GRID[i, j]*self.DELTA_TIME if steer_rate else self.STEER_GRID[i, j]
        return vel, steer
    
    def choose_exit(self, entry):
        exit_headings = np.array([h for h in self.EXIT_HEADINGS if np.abs(h - entry) < self.MAX_EXIT_DELTA])
        return np.random.choice(exit_headings)

    def init_mocap(self):
        def state_cb(pose, twist):
            state = VehicleStateMsg()
            state.header = pose.header
            state.child_frame_id = self.NAME
            state.x = pose.pose.position.x 
            state.y = pose.pose.position.y
            roll, pitch, yaw = euler_from_quaternion([pose.pose.orientation.x,
                                                      pose.pose.orientation.y,
                                                      pose.pose.orientation.z,
                                                      pose.pose.orientation.w])
            state.yaw = yaw
            state.v = twist.twist.linear.x
            self.state = state
            
        mf.TimeSynchronizer([
            mf.Subscriber(f'/qualisys/{self.NAME}/pose', PoseStamped),
            mf.Subscriber(f'/qualisys/{self.NAME}/velocity', TwistStamped)
        ], 10).registerCallback(state_cb)


    def run(self):

        ## Wait for init time
        rospy.sleep(self.INIT_WAIT)
        self.init_time = rospy.Time.now()

        ## Initialize sessions
        self.sessions = {}
        for id in range(self.NUM_SESSIONS):
            self.init_session(id)

        current_session_id = 0
        
        while not rospy.is_shutdown() and current_session_id < self.NUM_SESSIONS:
            # Apply control
            session = self.sessions[current_session_id]
        
            vel, steer = self.control_update(session['limits'], steer_rate=True if self.STATE_DIMS == 5 else False)
            self.steer = steer
            # Send control to vehicle
            pass
        
            
            # Status updates
            for id in range(self.NUM_SESSIONS):
                # Publish state
                if id >= current_session_id:
                    self.sessions[id]['states_pub'].publish(self.state)
                # Update entry time
                if id == current_session_id:
                    pass
                elif id == current_session_id + 1:
                    self.sessions[id]['earliest_entry'] = self.sessions[id-1]['earliest_exit']
                    self.sessions[id]['latest_entry'] = self.sessions[id-1]['latest_exit']
                    self.sessions[id]['entry_time'] = (self.sessions[id]['earliest_entry']+self.sessions[id]['latest_entry'])/2
                    notify_res = self.sessions[id]['notify_srv'](self.sessions[id]['entry_time'])
                    if not notify_res.success:
                        rospy.logerr(f'Failed to notify ITS: {notify_res.message}')
                    else:
                        self.sessions[id]['exit_time'] = self.sessions[id]['entry_time'] + rospy.Duration(notify_res.transit_time)
                    
                    
                    
                    
                    
            # Check if session is over
            pass
            current_session_id += 1

            # Sleep
            self.rate.sleep()

if __name__ == '__main__':

    ## Start node ##
    Vehicle().run()

