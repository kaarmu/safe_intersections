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

def around(l, e, n):
    if e not in l: return []
    i = l.index(e)
    N = len(l)
    return [l[(i+j) % N] for j in range(-n, n+1)]

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
    LOOP_TIME = 5.0
    
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
    ARRIVAL_TIMES = np.arange(0, NUM_SESSIONS*LOOP_TIME, LOOP_TIME)
    # EXIT_TIMES = np.arange(LOOP_TIME, NUM_SESSIONS*LOOP_TIME+LOOP_TIME, LOOP_TIME)
    
    LOCATIONS = [
        'center_e', 'center_ene', 'center_ne', 'center_nne',
        'center_n', 'center_nnw', 'center_nw', 'center_wnw',
        'center_w', 'center_wsw', 'center_sw', 'center_ssw',
        'center_s', 'center_ese', 'center_se', 'center_sse',
    ]
    PERMITTED_ROUTES = {
        (_entry, _exit): ('outside',)
        for _entry in LOCATIONS
        for _exit in set(LOCATIONS) - set(around(LOCATIONS, _entry, 4)) # flip
    }

    ENTRY_LOCATIONS = LOCATIONS
    EXIT_LOCATIONS = LOCATIONS
    LOCATIONS += ['outside']
    PERMITTED_ROUTES.update({
        ('init', _exit): ('outside',)
        for _exit in 'center_ne center_ene center_e'.split()
    })
    
    LIMITS_SHAPE = (50, 50)

    DEBUG = True

    def __init__(self):

        ## Initialize node

        rospy.init_node(self.__class__.__name__)

        ## Load parameters

        self.NAME = load_param('~name', 'svea')
        self.AREA = load_param('~area', 'sml')
        self.INIT_WAIT = load_param('~init_wait', 5)

        # rate
        self.rate = rospy.Rate(10)

        # initialize mocap and state
        self.init_mocap()
        self.steer = 0.0
        self.in_zone = False

        ## Create service proxies
        self.connect_srv = rospy.ServiceProxy('connect', Connect)

        ## Node initialized
        rospy.loginfo(f'{self.__class__.__name__} initialized!')
        
    def init_session(self, id):
        def limits_cb(msg):
            limits = np.frombuffer(msg.data, dtype=np.float32)
            self.sessions[id]['limits'] = limits.reshape(self.LIMITS_SHAPE)
            
        name = f'{self.NAME}_{id}'
        arrival_time = self.init_time + rospy.Duration(self.ARRIVAL_TIMES[id])
        ## Connect to LTMS
        resp = self.connect_srv(name, arrival_time)
        if not resp.success:
            rospy.logerr(f'Failed to connect to LTMS: {resp.message}')
            return
        ## Set up session
        self.sessions[id]['name'] = name
        self.sessions[id]['arrival_time'] = arrival_time
        self.sessions[id]['its_id'] = resp.its_id
        self.sessions[id]['states_pub'] = rospy.Publisher(f'{name}/states', VehicleStateMsg, queue_size=1)
        self.sessions[id]['notify_srv'] = rospy.ServiceProxy(f'{resp.its_id}/notify', Notify)
        self.sessions[id]['reserve_srv'] = rospy.ServiceProxy(f'{resp.its_id}/reserve', Reserve)
        self.sessions[id]['limits_sub'] = rospy.Subscriber(f'{resp.its_id}/limits', bytes, limits_cb)
        self.sessions[id]['valid'] = True
        ## Notify ITS
        notify_res = self.sessions[id]['notify_srv'](arrival_time)
        if not notify_res.success:
            rospy.logerr(f'Failed to notify ITS: {notify_res.message}')
            return
        self.sessions[id]['departure_time'] = arrival_time + rospy.Duration(notify_res.transit_time)
    
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
        permitted_exits = [exit for exit in self.EXIT_LOCATIONS if self.PERMITTED_ROUTES.get((entry, exit))]
        return np.random.choice(permitted_exits)

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
        
    def is_at_exit(self, zone=(0, 0), radius=0.25):
        if np.linalg.norm([self.state.x - zone[0], self.state.y - zone[1]]) < radius and not self.in_zone:
            self.in_zone = True
            return True
        else:
            return False
    
    def update_zone_flag(self, zone=(0, 0), radius=0.5):
        if np.linalg.norm([self.state.x - zone[0], self.state.y - zone[1]]) > radius and self.in_zone:
            self.in_zone = False
        
    def reserve_entry_exit(self, id):
        name = self.sessions[id]['name']
        time_ref = rospy.Time.now()
        earliest_entry = time_ref if id == 0 else self.sessions[id-1]['earliest_exit']
        latest_entry = time_ref + self.LOOP_TIME if id == 0 else self.sessions[id-1]['latest_exit']
        entry = 'init' if id == 0 else self.sessions[id-1]['exit_location']
        exit = self.choose_exit(entry)
        # Fill session
        self.sessions[id]['entry'] = entry
        self.sessions[id]['exit'] = exit
        self.sessions[id]['earliest_entry'] = res.earliest_entry
        self.sessions[id]['latest_entry'] = res.latest_entry
        # Reserve request
        req = Reserve.Request(name, entry, exit, time_ref, earliest_entry-time_ref, latest_entry-time_ref)
        res = self.sessions[id]['reserve_srv'](req)
        if res.success:
            self.sessions[id]['earliest_exit'] = res.time_ref + res.earliest_exit
            self.sessions[id]['latest_exit'] = res.time_ref + res.latest_exit
            return True
        else:
            rospy.logerr(f'Failed to reserve entry/exit: {res.message}')           
            return False
    
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
            if current_session_id == 0:
                ## Assume vehicle has already entered the area
                self.reserve_entry_exit(current_session_id)
            else:
                ## check if vehicle is at entry
                if self.is_at_exit():
                    current_session_id += 1
                    self.reserve_entry_exit(current_session_id)
            
            # Apply control
            if self.sessions[current_session_id]['limits']:        
                vel, steer = self.control_update(self.sessions[current_session_id]['limits'], steer_rate=True if self.STATE_DIMS == 5 else False)
                self.steer = steer
                # Send control to vehicle
                pass
        
            
            # Status updates
            for id in range(self.NUM_SESSIONS):
                if id >= current_session_id:
                    # Publish state
                    self.sessions[id]['states_pub'].publish(self.state)
                if id > current_session_id:
                    # Update arrival time and notify ITS
                    self.sessions[id]['arrival_time'] = self.sessions[id-1]['departure_time']
                    notify_res = self.sessions[id]['notify_srv'](self.sessions[id]['arrival_time'])
                    if not notify_res.success:
                        rospy.logerr(f'Failed to notify ITS: {notify_res.message}')
                        return
                    self.sessions[id]['departure_time'] = self.sessions[id]['arrival_time'] + rospy.Duration(notify_res.transit_time)
                    
            # Update zone flag
            self.update_zone_flag()

            # Sleep
            self.rate.sleep()

if __name__ == '__main__':

    ## Start node ##
    Vehicle().run()

