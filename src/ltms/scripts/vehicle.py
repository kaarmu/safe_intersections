#! /usr/bin/env python3

import numpy as np
import rospy
from collections import deque

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

def publish_initialpose(state, n=10):
    p = PoseWithCovarianceStamped()
    p.header.frame_id = "map"
    p.pose.pose.position.x = state.x
    p.pose.pose.position.y = state.y

    q = quaternion_from_euler(0, 0, state.yaw)
    p.pose.pose.orientation.z = q[2]
    p.pose.pose.orientation.w = q[3]

    pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=10)
    rate = rospy.Rate(10)

    for _ in range(n):
        pub.publish(p)
        rate.sleep()


class Vehicle:

    DELTA_TIME = 0.1

    OUTSIDE_ROUTES = {
        'exit_s': 'entry_w',
        'exit_w': 'entry_n',
        'exit_n': 'entry_e',
        'exit_e': 'entry_s',
    }

    INSIDE_ROUTES = {
        'entry_s': ['exit_w', 'exit_n', 'exit_e'],
        'entry_w': ['exit_n', 'exit_e', 'exit_s'],
        'entry_n': ['exit_e', 'exit_s', 'exit_w'],
        'entry_e': ['exit_s', 'exit_w', 'exit_n'],
    }

    def __init__(self):

        ## Initialize node

        rospy.init_node(self.__class__.__name__)

        ## Load parameters

        self.NAME = load_param('~name', 'svea')
        self.AREA = load_param('~area', 'sml')
        self.STATE = load_param("~state", [4, 4, np.pi*3/4, 0])
        self.IS_SIM = load_param('~is_sim', True)
        self.USE_MOCAP = load_param('~use_mocap', False)

        ## Set initial values

        # initial state
        state = VehicleState(*self.STATE)
        publish_initialpose(state)

        self.goto_waypoints = deque(maxlen=5)

        ## Create simulators, models, managers, etc.

        self.rate = rospy.Rate(10)

        if self.IS_SIM:
            # simulator need a model to simulate
            self.sim_model = SimpleBicycleModel(state)

            # start the simulator immediately, but paused
            self.simulator = SimSVEA(
                self.sim_model, dt=self.DELTA_TIME, run_lidar=True, start_paused=True
            ).start()

        ## Start lli interfaces

        if self.USE_MOCAP:
            self.init_mocap()

        self.localization = LocalizationInterface().start()
        self.actuation = ActuationInterface().start(wait=True)
        self.planning = PlannerInterface()

        self.state = self.localization.state

        ## Create controller and planner

        self.planner = AStarPlanner(0.1, limit=[[-5, 5], [-5, 5]], #obstacles=centered_rect(2.5, 2.5, 0.6, 0.5)
                                    )
        self.controller = PurePursuitController()
        self.controller.set_path([[0, 0]])

        ## Create transformer

        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()

        ## Create publishers

        ## Create subscribers

        rospy.Subscriber('/clicked_point', PointStamped, self.clicked_point_cb)

        ## Node initialized

        rospy.loginfo(f'{self.__class__.__name__} initialized!')

        # everything ready to go -> unpause simulator
        if self.IS_SIM:
            self.simulator.toggle_pause_simulation()

    def init_mocap(self):
        self._state_pub = rospy.Publisher('state', VehicleStateMsg, latch=True, queue_size=1)
        def state_cb(pose, twist):
            state = VehicleStateMsg()
            state.header = pose.header
            state.child_frame_id = 'svea2'
            state.x = pose.pose.position.x 
            state.y = pose.pose.position.y
            roll, pitch, yaw = euler_from_quaternion([pose.pose.orientation.x,
                                                        pose.pose.orientation.y,
                                                        pose.pose.orientation.z,
                                                        pose.pose.orientation.w])
            state.yaw = yaw
            state.v = twist.twist.linear.x
            self._state_pub.publish(state)
        mf.TimeSynchronizer([
            mf.Subscriber(f'/qualisys/{self.NAME}/pose', PoseStamped),
            mf.Subscriber(f'/qualisys/{self.NAME}/velocity', TwistStamped)
        ], 10).registerCallback(state_cb)

    def goto(self, goal):
        init = self.state.x, self.state.y
        path = self.planner.plan(init, goal)
        self.planning.set_points_path(path)
        self.planning.publish_path()
        self.controller.set_path(path)
        self.controller.set_target_velocity(0.8)
        self.controller.is_finished = False
        rospy.loginfo("Going to (%f, %f)", *goal)
        return path
    
    def goto_next_waypoint(self):
        if self.goto_waypoints:
            goal = self.goto_waypoints.popleft()
            self.goto(goal)
    
    def clicked_point_cb(self, msg):
        wayp = msg.point.x, msg.point.y
        if wayp not in self.planner.world:
            rospy.loginfo("Waypoint not in world.")
        self.goto_waypoints.append(wayp)
        
    def spin(self):

        if self.controller.is_finished:
            self.goto_next_waypoint()

        steering, velocity = self.controller.compute_control(self.state)
        self.actuation.send_control(steering, velocity)

    def run(self):
        while not rospy.is_shutdown():
            self.spin()
            self.rate.sleep()


### dirty tests

def plot_obs(obs):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    for x, y, r in obs:
        patch = plt.Circle((x, y), r, fill=True, color='black')
        ax.add_patch(patch)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.show()
def centered_rect(w, h, r, d):
    p1 = -w/2, -h/2
    p2 = -w/2, +h/2
    p3 = +w/2, +h/2
    p4 = +w/2, -h/2
    return rect([p1, p2, p3, p4], r, d)
def rect(ps, r, d):
    p1, p2, p3, p4 = ps
    out = []
    out.extend(line(p1, p2, r, d))
    out.extend(line(p2, p3, r, d))
    out.extend(line(p3, p4, r, d))
    out.extend(line(p4, p1, r, d))
    return out
def line(p1, p2, r, d):
    n = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]) / d)
    out = np.zeros((n+1, 3))
    out[:, 0] = np.linspace(p1[0], p2[0], n+1)
    out[:, 1] = np.linspace(p1[1], p2[1], n+1)
    out[:, 2] = r
    return out
def test_obs():
    plot_obs(centered_rect(2.5, 2.5, 0.5, 0.5))

if __name__ == '__main__':

    ## Start node ##
    Vehicle().run()

