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
from ltms.srv import ReserveCorridor
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

def pathlen(path):
    dist = 0
    for i, nxt in enumerate(path[1:]):
        pnt = path[i]
        dist += np.hypot(nxt[0] - pnt[0], nxt[1] - pnt[1])
    return dist

def service_like(f):
    q = SimpleQueue()
    def worker():
        while not rospy.is_shutdown():
            try:
                item = q.get(timeout=1)
            except Exception: pass
            else:
                args, kwds = item
                f(*args, **kwds)
    t = Thread(target=worker)
    def wrapper(*args, **kwds):
        q.put((args, kwds))
        if not t.is_alive():
            t.start()
    return wrapper

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

    # Enable U-turns
    INSIDE_ROUTES['entry_s'].append('exit_s')
    INSIDE_ROUTES['entry_w'].append('exit_w')
    INSIDE_ROUTES['entry_n'].append('exit_n')
    INSIDE_ROUTES['entry_e'].append('exit_e')

    LOCATION_COORDINATES = {
        'entry_s': (+0.25, -1.00),
        'entry_w': (-1.00, -0.25),
        'entry_n': (-0.25, +1.00),
        'entry_e': (+1.00, +0.25),
        'exit_s':  (-0.25, -1.00), 
        'exit_w':  (-1.00, +0.25), 
        'exit_n':  (+0.25, +1.00), 
        'exit_e':  (+1.00, -0.25), 
    }

    MAX_VEL = 0.8
    MIN_VEL = 0.4
    MED_VEL = (MAX_VEL + MIN_VEL)/2

    DEBUG = True

    def __init__(self):

        ## Initialize node

        rospy.init_node(self.__class__.__name__)

        ## Load parameters

        self.NAME = load_param('~name', 'svea')
        self.AREA = load_param('~area', 'sml')
        self.INIT_STATE = load_param("~init_state", [4, 4, np.pi*3/4, 0])
        self.INIT_ENTRY = load_param('~init_entry', 'entry_e')
        self.INIT_WAIT = load_param('~init_wait', 5)
        self.IS_SIM = load_param('~is_sim', True)
        self.USE_MOCAP = load_param('~use_mocap', False)

        ## Set initial values

        # initial state
        state = VehicleState(*self.INIT_STATE)
        publish_initialpose(state)

        self.goto_waypoints = deque(maxlen=5)
        self.paths = deque()

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

        self.planner = AStarPlanner(0.1, limit=[[-5, 5], [-5, 5]], obstacles=create_4way_obs())
        self.controller = PurePursuitController()
        self.controller.set_path([[0, 0]])

        ## Create transformer

        self.buffer = tf2_ros.Buffer(rospy.Duration(10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.br = tf2_ros.TransformBroadcaster()

        ## Create service proxies

        # self.req_corridor = rospy.ServiceProxy('/ltms/request_corridor', ReserveCorridor)

        ## Create publishers

        ## Create subscribers

        rospy.Subscriber('/clicked_point', PointStamped, self.clicked_point_cb)

        ## Node initialized

        rospy.loginfo(f'{self.__class__.__name__} initialized!')

        # everything ready to go -> unpause simulator
        if self.IS_SIM:
            self.simulator.toggle_pause_simulation()

        self.localization.block_until_state()

        start_time = datetime.now() + timedelta(seconds=self.INIT_WAIT)

        clovers = np.load(Path(__file__) / '../data/clovers-v1.npz', allow_pickle=True)

        self.append_path(clovers['exit_w'])
        self.append_path(clovers['exit_s'])
        self.append_path(clovers['exit_e'])
        self.append_path(clovers['exit_n'])

        self.INIT_ENTRY = ''
        if self.INIT_ENTRY:

            # path to entry
            time_ref = start_time
            entry_loc = self.INIT_ENTRY
            init_path = self.plan_to_location(entry_loc)
            init_path_dist = pathlen(init_path)
            earliest_entry = init_path_dist / self.MAX_VEL
            latest_entry = init_path_dist / self.MIN_VEL
            entry_loc = random.choice(list(self.INSIDE_ROUTES)) 
            resp, exit_loc, outside_path, reentry_loc = self.plan_inside_outside(time_ref, entry_loc, earliest_entry, latest_entry)
            
            assert resp.success, 'Not successful planning'
            entry_time = datetime.fromisoformat(resp.time_ref) 
            entry_time += timedelta(seconds=resp.earliest_entry+resp.latest_entry) / 2

            init_vel = init_path_dist / (entry_time - start_time).total_seconds()
            print('init_vel:', init_vel)
            self.append_path(init_path, init_vel)

            # entry to reentry
            inside_init = self.LOCATION_COORDINATES[entry_loc]
            inside_goal = self.LOCATION_COORDINATES[exit_loc]
            inside_corridor = np.frombuffer(bytes(resp.corridor), bool).reshape(resp.shape)
            inside_path = self.one_shot_plan(inside_init, inside_goal, ~inside_corridor)
            self.append_path(inside_path)

            self.append_path(outside_path)
            
        if False:
            xy = (self.state.x, self.state.y)
            path = self.plan_to_location(xy, self.INIT_ENTRY)
            dist = pathlen(path)
            
            while not rospy.is_shutdown():
                _ = self.sched_route(time_ref=start_time,
                                     entry_loc=self.INIT_ENTRY,
                                     earliest_entry=round(dist/self.MAX_VEL, 1),
                                     latest_entry=round(dist/self.MIN_VEL, 1),
                                     exit_loc=random.choice(self.INSIDE_ROUTES[self.INIT_ENTRY]))
                if ...:
                    break
                else:
                    start_time += timedelta(seconds=2)
        
        now = datetime.now()
        while not rospy.is_shutdown() and now < start_time:
            rospy.sleep(0.02)
            now = datetime.now()
            print(f'{(start_time - now).total_seconds():0.02f} s left until start.', end='\r', flush=True)
        
        def switching_worker():
            rate = rospy.Rate(10)
            switch = True
            while not rospy.is_shutdown():
                f = self.outside_spin if switch else self.inside_spin
                switch ^= f()
                rate.sleep()
                
        # Thread(target=switching_worker).start()

        print(f'Starting!'.ljust(25))

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

    def plan_inside_outside(self, time_ref, entry_loc, earliest_entry, latest_entry):
        exit_loc = random.choice(self.INSIDE_ROUTES[entry_loc])

        req = ReserveCorridor._request_class(
            name=self.NAME,
            entry=entry_loc,
            exit=exit_loc,
            time_ref=time_ref.isoformat(),
            earliest_entry=earliest_entry,
            latest_entry=latest_entry,
        )

        t0 = time()
        resp = self.req_corridor(req)
        t1 = time()

        outside_path, reentry_loc = self.plan_outside(exit_loc)
        outside_dist = pathlen(outside_path)

        if self.DEBUG:
            print('[req]')
            print(req)
            print('[resp]')
            print(resp)
            print('[time]')
            print('latency:', t1-t0)
            print('now:', datetime.now().isoformat())
            print('[path]')
            print('state:', self.state.x, self.state.y)
            print('goal:', *outside_path[-1])
            print('steps:', len(outside_path))
            print('dist:', outside_dist)
            print('earliest:', outside_dist/self.MAX_VEL)
            print('latest:', outside_dist/self.MIN_VEL)

        return resp, exit_loc, outside_path, reentry_loc

    def plan_outside(self, entry_loc):
        exit_loc = self.OUTSIDE_ROUTES[entry_loc]
        entry_xy = self.LOCATION_COORDINATES[entry_loc]
        exit_xy = self.LOCATION_COORDINATES[exit_loc]
        path = self.planner.plan(entry_xy, exit_xy)
        return path, exit_loc
    
    def one_shot_plan(self, init, goal, occ_grid):
        planner = AStarPlanner(2.5 / 31, limit=[[-1.25, 1.25], [-1.25, 1.25]], occ_grid=occ_grid)        
        return planner.plan(init, goal)

    def plan_to_goal(self, goal):
        init = self.state.x, self.state.y
        return self.planner.plan(init, goal)

    def plan_to_location(self, loc):
        xy = self.LOCATION_COORDINATES[loc]
        return self.plan_to_goal(xy)

    def goto(self, goal):
        path = self.plan_to_goal(goal)
        if path:
            self.follow_path(path)
            rospy.loginfo("Going to (%f, %f)", *goal)
        else:
            rospy.loginfo('Could not find a path to (%f, %f)', *goal)
        return path

    def follow_path(self, path, vel=None):
        self.planning.set_points_path(path)
        self.planning.publish_path()
        self.controller.set_path(path)
        self.controller.set_target_velocity(self.MED_VEL if vel is None else vel)
        self.controller.is_finished = False
    def append_path(self, *args, **kwds):
        self.paths.append((args, kwds))
    def next_path(self):
        args, kwds = self.paths.popleft()
        self.follow_path(*args, **kwds)

    def goto_next_waypoint(self):
        if self.goto_waypoints:
            goal = self.goto_waypoints.popleft()
            self.goto(goal)
    
    def clicked_point_cb(self, msg):
        wayp = msg.point.x, msg.point.y
        if wayp not in self.planner.world:
            rospy.loginfo("Waypoint not in world.")
        self.goto_waypoints.append(wayp)
        
    def inside_spin(self):
        
        # ask for lrcs
        lrcs = ...
        

    def outside_spin(self):

        
        return False

    def run(self):

        rate = rospy.Rate(10)

        t0 = time()

        while not rospy.is_shutdown() and not (self.controller.is_finished and not self.paths):

            if self.controller.is_finished:
                t1 = time()
                print('path completed in:', t1 - t0)
                self.next_path()
                t0 = t1

            steering, velocity = self.controller.compute_control(self.state)
            self.actuation.send_control(steering, velocity)

            rate.sleep()

        print('finished')
        rospy.spin()



### dirty tests

def plot_obs(obs):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    # bgim = plt.imread('/home/kaarmu/Projects/safe_intersection/src/ltms/data/background.png')
    ax.imshow(bgim, extent=[-1.25, 1.25, -1.25, 1.25])
    for x, y, r in obs:
        patch = plt.Circle((x, y), r, fill=True, color='black')
        ax.add_patch(patch)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    fig.show()
def line(p1, p2, r, d):
    n = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]) / d)
    out = np.zeros((n+1, 3))
    out[:, 0] = np.linspace(p1[0], p2[0], n+1)
    out[:, 1] = np.linspace(p1[1], p2[1], n+1)
    out[:, 2] = r
    return out
def shape(ps, r, d):
    return np.concatenate([
        line(ps[i], ps[(i+1) % len(ps)], r, d)
        for i in range(len(ps))
    ])
def translate(xy, s):
    s = np.array(s)
    s[:, :2] += xy
    return s
def rect(w, h, r, d):
    p1 = -w/2, -h/2
    p2 = -w/2, +h/2
    p3 = +w/2, +h/2
    p4 = +w/2, -h/2
    return shape([p1, p2, p3, p4], r, d)
def create_4way_obs():
    corner = lambda xy: translate(xy, rect(w=0.5, h=0.5, r=0.25, d=0.25))
    tl, tr, bl, br = map(corner, [(-1, +1), (-1, -1), (+1, -1), (+1, +1)])
    c = rect(1, 1, r=0.1, d=0.15)
    return np.concatenate((tl, tr, bl, br, c))


if __name__ == '__main__':

    ## Start node ##
    Vehicle().run()

