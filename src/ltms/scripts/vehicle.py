#! /usr/bin/env python3

import numpy as np
import random
import secrets
from time import time
from collections import deque
from datetime import datetime, timedelta
from threading import Thread
from queue import SimpleQueue

# SVEA imports
from svea.interfaces import ActuationInterface
from svea_msgs.msg import VehicleState as VehicleStateMsg

# ROS imports
import rospy
from ltms.msg import NamedBytes
from ltms.srv import Connect, Notify, Reserve
from ltms_util.debuggable_lock import *
import message_filters as mf
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, TwistStamped, PointStamped
from nav_msgs.msg import Path
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from nats_ros_connector.nats_manager import NATSManager
from threading import RLock
from contextlib import contextmanager

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

_LOCATIONS = ['left', 'top', 'right', 'bottom']
_PERMITTED_ROUTES = {
    (_entry, _exit): ('full',)
    for _entry in _LOCATIONS
    for _exit in set(_LOCATIONS) - {_entry, _LOCATIONS[_LOCATIONS.index(_entry)-1]}
}

class Vehicle:
    
    NUM_SESSIONS = 5
    MAX_SESSIONS = 50

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
    
    STATE_DIMS = 5 # or 5 if including steering rate
    
    if STATE_DIMS == 4:
        U1_VEC = np.linspace(MIN_STEER, MAX_STEER)
        U2_VEC = np.linspace(MIN_ACC, MAX_ACC)
    elif STATE_DIMS == 5:
        U1_VEC = np.linspace(MIN_STEER_RATE, MAX_STEER_RATE)
        U2_VEC = np.linspace(MIN_ACC, MAX_ACC)

    U1_GRID, U2_GRID = np.meshgrid(U1_VEC, U2_VEC)
    
    ENTRY_LOCATIONS = _LOCATIONS + ['init']
    EXIT_LOCATIONS = _LOCATIONS
    LOCATIONS = _LOCATIONS + ['full']
    PERMITTED_ROUTES = dict(list(_PERMITTED_ROUTES.items()) + [
        (('init', _exit), ('full',))
        for _exit in 'left'.split()
    ])

    
    LIMITS_SHAPE = (50, 50)

    DEBUG = True

    def __init__(self):

        ## Initialize node

        rospy.init_node(self.__class__.__name__, log_level=rospy.DEBUG)

        ## Load parameters

        self.NAME = load_param('~name', 'svea')
        self.AREA = load_param('~area', 'sml')

        self.INIT_WAIT = load_param('~init_wait', 20)
        self.INIT_WAIT = timedelta(seconds=self.INIT_WAIT)

        self.RES_TIME_LIMIT = load_param('~res_time_limit', 30)
        self.RES_TIME_LIMIT = timedelta(seconds=self.RES_TIME_LIMIT)

        ## Session management

        self.sessions_lock = debuggable_lock('sessions_lock', RLock())
        self.sessions = {}
        self.session_order = []

        self.reserve_q = SimpleQueue()

        def worker():
            while not rospy.is_shutdown():
                sid = self.reserve_q.get()
                for sess in self.select_session(sid):
                    if sess['reserved']: return # just double check
                    info = self.reserve(sess)
                    if not info['success']:
                        # Justify arrival time and notify server
                        arrival_time = sess['arrival_time'] + timedelta(seconds=1)
                        resp = self.notify_srv(sess['sid'], arrival_time.isoformat())
                        sess['arrival_time'] = arrival_time
                        sess['departure_time'] = arrival_time + timedelta(seconds=resp.transit_time)
                else:
                    # session that is marked-for-reservation were cleaned up
                    assert False, 'Undefined Behavior'


        Thread(target=worker).start()

        ## Rate
        
        self.rate = rospy.Rate(10)

        ## Initialize mocap
        
        self.state = VehicleStateMsg()

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

        ## Initiatlize interfaces

        self.actuator = ActuationInterface(self.NAME).start()
        self.nats_mgr = NATSManager()

        ## Create service proxies
        self.connect_srv = self.nats_mgr.new_serviceproxy('/server/connect', Connect)
        self.notify_srv = self.nats_mgr.new_serviceproxy('/server/notify', Notify)
        self.reserve_srv = self.nats_mgr.new_serviceproxy('/server/reserve', Reserve)
        
        self.target = None
        def limits_cb(msg):
            for sess in self.select_session(msg.name):
                rospy.logdebug('\n'.join([
                    f'Received new driving limits.',
                    f'  name: {msg.name}',
                    f'  delta: {(rospy.Time.now() - msg.stamp).to_sec()} s',
                ]))
                limits = np.frombuffer(bytes([x+128 for x in msg.mask]), dtype=bool)
                sess['limits'] = limits.reshape(len(self.U1_VEC), len(self.U2_VEC))

        self.State = self.nats_mgr.new_publisher(f'/server/state', VehicleStateMsg, queue_size=5)
        self.Limits = self.nats_mgr.new_subscriber(f'/server/limits', NamedBytes, limits_cb)

        def limits_tmr(event):
            sid = self.session_order[0]
            self.state.child_frame_id = sid
            self.State.publish(self.state)
        rospy.Timer(rospy.Duration(0.5), limits_tmr)

        ## Node initialized
        rospy.loginfo(f'{self.__class__.__name__} initialized!')
    
    def choose_exit(self, entry):
        permitted_exits = [exit for exit in self.EXIT_LOCATIONS if (entry, exit) in self.PERMITTED_ROUTES]
        return np.random.choice(permitted_exits)

    @property
    def iter_sessions(self):
        with self.sessions_lock('iter_sessions'):
            for sid, sess in self.sessions.items():
                with sess['lock']('iter_sessions'):
                    yield (sid, sess)

    def sessions_add(self, key, dict1=None, **kwds):
        assert bool(dict1 is not None) ^ bool(kwds), 'Use either dict1 or keywords'
        with self.sessions_lock('sessions_add'):
            self.sessions[key] = kwds if dict1 is None else dict1

    def select_session(self, sid):
        with self.sessions_lock('select_session'):
            if sid not in self.sessions: return
            with self.sessions[sid]['lock']('select_session'):
                yield self.sessions[sid]

    def create_session(self, parent_sid=None, arrival_time=None, entry_loc=None, exit_loc=None):
        
        ## Get entry information
        if parent_sid is not None:
            for parent_sess in self.select_session(parent_sid):
                arrival_time = parent_sess['departure_time']
                entry_loc = parent_sess['exit_loc']
                break
            else:
                # Parent doesn't exist
                assert False, 'Undefined Behavior'

        ## Make up a destination
        exit_loc = self.choose_exit(entry_loc)

        ## Create new random session id
        sid = f'{self.NAME}_{secrets.token_hex(4)}'

        ## Make sure we have time, entry, exit

        assert arrival_time is not None
        assert entry_loc is not None
        assert exit_loc is not None

        ## Connect to LTMS
        resp = self.connect_srv(sid, arrival_time.isoformat())
        transit_time = resp.transit_time
        latest_reserve_time = resp.latest_reserve_time

        ## Set up session
        self.sessions_add(sid,
                          sid=sid,
                          lock=debuggable_lock(f'{sid}_lock', RLock()),
                          parent=parent_sess,
                          entry_loc=entry_loc,
                          exit_loc=exit_loc,
                          reserved=False,
                          limits=None,
                          arrival_time=arrival_time,
                          latest_reserve_time=latest_reserve_time,
                          departure_time=arrival_time + timedelta(seconds=transit_time))

        return sid

    def sessions_update(self):
        now = datetime.now()

        with self.sessions_lock('sessions_update'):
            assert self.session_order, 'We always assume that when sessions_update is run, there is some prev session'

            active_sid = self.session_order[0]

            # clean up sessions list
            for sid in self.sessions:
                is_upcomming = sid in self.session_order
                is_parent = sid == self.sessions[active_sid]['parent']['sid']
                if not (is_upcomming or is_parent):
                    del self.sessions[sid]

            # Mark ready-to-reserve
            for sid in list(self.session_order):
                sess = self.sessions[sid]
                if sess['reserved']: continue
                if sess['latest_reserve_time'] - now <= self.RES_TIME_LIMIT:
                    self.reserve_q.put(sid)

            for i in range(self.NUM_SESSIONS):
                # Update un-reserved sessions. Reserved sessions are static.
                if i < len(self.session_order):
                    sid = self.session_order[i]
                    for sess in self.select_session(sid):
                        if not sess['reserved']:
                            resp = self.notify_srv(sid, sess['parent']['earliest_exit'])
                            sess['latest_reserve_time'] = datetime.fromisoformat(resp.latest_reserve_time)
                            sess['arrival_time'] = sess['parent']['departure_time']
                            sess['departure_time'] = sess['arrival_time'] + timedelta(seconds=resp.transit_time)

                # Append new sessions
                else:
                    parent_sid = self.session_order[i-1]
                    sid = self.create_session(parent_sid)
                    self.session_order.append(sid)

            rospy.logdebug(f'# Reserve Queue Length: {self.reserve_q.qsize()}')

    def reserve(self, sess):
        # Assumes sess is already locked by caller
        rospy.logdebug('Reserving for %s', sess['sid'])
        start_time = datetime.now()

        if sess['reserved']: 
            rospy.logdebug('> Already reserved!', sess['sid'])
            return {'success': True}

        resp = self.reserve_srv(name=sess['sid'],
                                entry=sess['entry_loc'],
                                exit=sess['exit_loc'],
                                time_ref=sess['arrival_time'].isoformat(),
                                earliest_entry=0,
                                latest_entry=3) # Let server choose when exactly to enter
    
        if not resp.success:
            rospy.loginfo('Reservation for %s failed: %s', sess['sid'], resp.reason)
            return {'success': False}
        
        rospy.logdebug(f'> Reservation succeeded! Took: {datetime.now() - start_time}')

        sess['time_ref'] = datetime.fromisoformat(resp.time_ref)
        sess['earliest_entry'] = resp.earliest_entry
        sess['latest_entry'] = resp.latest_entry
        sess['earliest_exit'] = resp.earliest_exit
        sess['latest_exit'] = resp.latest_exit
    
        sess['arrival_time'] = sess['time_ref'] 
        sess['arrival_time'] += timedelta(seconds=(sess['earliest_entry'] + sess['latest_entry'])/2)
        sess['departure_time'] = sess['time_ref']
        sess['departure_time'] += timedelta(seconds=(sess['earliest_exit'] + sess['latest_exit'])/2)

        sess['reserved'] = True

        rospy.logdebug('> Reservation complete for %s', sess['sid'])
        return {'success': True}

    def run(self):

        ## Wait for init time
        start_time = datetime.now() + self.INIT_WAIT

        ## Initialize sessions
        with self.sessions_lock('run [Initialize sessions]'):
            entry_loc = 'init'
            exit_loc = self.choose_exit(entry_loc)
            init_sid = self.create_session(None, start_time, entry_loc, exit_loc)
            self.session_order.append(init_sid)
        rospy.Timer(rospy.Duration(1), lambda event: self.sessions_update())

        ## Wait for start time
        while not rospy.is_shutdown():
            if datetime.now() >= start_time:
                break
            rospy.sleep(0.1)

        steering, velocity = 0, 0

        active_sid = init_sid
        rospy.loginfo('Initial session ID: %s', active_sid)

        while not rospy.is_shutdown():

            with self.sessions_lock('run [Update session order & Get Limits]'):
                
                if self.sessions[active_sid]['departure_time'] <= datetime.now():
                    self.session_order = self.session_order[1:]
                    active_sid = self.session_order[0]
                    rospy.loginfo('Switching to session ID: %s', active_sid)
            
                limits_mask = self.sessions[active_sid]['limits']

            if limits_mask is None:
                rospy.logwarn('Missing driving limits')
            else:
                # Update control
                if self.STATE_DIMS == 4:
                    limits_mask &= (self.U1_GRID >= steering + self.MIN_STEER_RATE)
                    limits_mask &= (self.U1_GRID <= steering + self.MAX_STEER_RATE)
                    args = np.argwhere(limits_mask)
                    if len(args) == 0:
                        rospy.loginfo('No admissible driving limits')
                    else:
                        n = np.random.randint(0, len(args))
                        i, j = args[n]
                        steering = self.U1_VEC[i]
                        velocity += self.U2_VEC[j] * self.DELTA_TIME
                if self.STATE_DIMS == 5:
                    args = np.argwhere(limits_mask)
                    if len(args) == 0:
                        rospy.loginfo('No admissible driving limits')
                    else:
                        n = np.random.randint(0, len(args))
                        i, j = args[n]
                        steering += self.U1_VEC[i] * self.DELTA_TIME
                        velocity += self.U2_VEC[j] * self.DELTA_TIME

            rospy.loginfo(f'Steering: {steering}, Velocity: {velocity}')
            self.actuator.send_control(steering=steering, velocity=velocity)

            # Sleep
            self.rate.sleep()

if __name__ == '__main__':

    ## Start node ##
    vehicle = Vehicle()
    vehicle.run()

