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
    
    NUM_SESSIONS_AHEAD = 4

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
        for _exit in 'bottom'.split()
    ])

    
    LIMITS_SHAPE = (50, 50)

    DEBUG = True

    def __init__(self):

        ## Initialize node

        rospy.init_node(self.__class__.__name__, log_level=rospy.DEBUG)

        ## Load parameters

        self.NAME = load_param('~name', 'svea')
        self.AREA = load_param('~area', 'sml')

        self.INIT_WAIT = load_param('~init_wait', 25)
        self.INIT_WAIT = timedelta(seconds=self.INIT_WAIT)

        self.RES_TIME_LIMIT = load_param('~res_time_limit', 30)
        self.RES_TIME_LIMIT = timedelta(seconds=self.RES_TIME_LIMIT)

        ## Session management

        self.sessions_lock = debuggable_lock('sessions_lock', RLock())
        self.sessions = {}
        
        self.active_sid = None
        self.session_order = []

        def cleaner_tmr(event):
            iter_opts = dict(name='cleaner',
                             skip=(self.active_sid, *self.session_order),
                             order=False,
                             lock_all=True)
            for sid in self.iter_sessions(**iter_opts):
                del self.sessions[sid]
                self.reserved -= {sid} # remove if it is reserved
        rospy.Timer(rospy.Duration(10), cleaner_tmr)

        self.reserved = set()
        self.reserving_sid = None

        def reserver():
            iter_opts = dict(name='reserver',
                             skip=self.reserved, 
                             order=True, 
                             lock_all=False)
                    
            for sid, sess in self.iter_sessions(**iter_opts):
                now = datetime.utcnow()
                if sess['latest_reserve_time'] - now <= self.RES_TIME_LIMIT:

                    self.reserving_sid = sid

                    success = False
                    while not success:
                        if rospy.is_shutdown(): return

                        success = self.reserve(sid)
                        if success:
                            self.reserved |= {sid} # add as reserved
                            return
                        else:
                            rospy.logwarn('\n'.join([
                                'Reservation unsuccessful!',
                                f'  Name:       {sid}',
                                f'  Time now:   {datetime.utcnow()}',
                            ]))
                            assert False, "Don't know what to do here yet"

                    self.reserving_sid = None

        rospy.Timer(rospy.Duration(1), lambda event: reserver())

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
            for sess in self.select_session(msg.name, lock_all=False, name='limits_cb'):
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
            if self.active_sid is not None:
                self.state.child_frame_id = self.active_sid
                self.State.publish(self.state)
        rospy.Timer(rospy.Duration(0.1), limits_tmr)

        ## Node initialized
        rospy.loginfo(f'{self.__class__.__name__} initialized!')
    
    def choose_exit(self, entry):
        permitted_exits = [exit for exit in self.EXIT_LOCATIONS if (entry, exit) in self.PERMITTED_ROUTES]
        return np.random.choice(permitted_exits)

    def iter_sessions(self, *, order=False, lock_all=False, skip=(), name=''):
        if name: name += ':'
        name += 'iter_sessions'

        def iteration(items):
            for sid, sess in items:
                if sid in skip: continue
                with sess['lock'](name):
                    yield (sid, sess)
            
        if lock_all:
            with self.sessions_lock(name):
                if order:
                    items = [(sid, self.sessions[sid]) 
                             for sid in self.session_order]
                else:
                    items = list(self.sessions.items())
                yield from iteration(items)
        else:
            with self.sessions_lock(name):
                items = list(self.sessions.items())
            yield from iteration(items)


    def sessions_add(self, key, dict1=None, **kwds):
        assert bool(dict1 is not None) ^ bool(kwds), 'Use either dict1 or keywords'
        with self.sessions_lock('sessions_add'):
            self.sessions[key] = kwds if dict1 is None else dict1

    def select_session(self, sid, *, strict=False, lock_all=True, name=''):
        if name: name += ':'
        name += 'select_session'
        if lock_all:
            with self.sessions_lock(name):
                if strict and sid not in self.sessions: raise Exception(f'Session {sid} not found!')
                if sid not in self.sessions: return
                with self.sessions[sid]['lock'](name):
                    yield self.sessions[sid]
        else:
            with self.sessions_lock(name):
                if strict and sid not in self.sessions: raise Exception(f'Session {sid} not found!')
                if sid not in self.sessions: return
                sess = self.sessions[sid]
            with sess['lock'](name):
                yield self.sessions[sid]

    def read_session_prop(self, sid, name):
        for sess in self.select_session(sid):
            return sess[name]

    def create_session_after_parent(self, parent_sid):

        for parent_sess in self.select_session(parent_sid, strict=True, name='create_session_after_parent'):
            arrival_time = parent_sess['departure_time']
            entry_loc = parent_sess['exit_loc']

        exit_loc = self.choose_exit(entry_loc)

        return self.create_session(arrival_time, entry_loc, exit_loc, parent=parent_sess)

    def create_session(self, arrival_time, entry_loc, exit_loc, parent=None):
        with self.sessions_lock('create_session'):

            ## Create new random session id
            sid = f'{self.NAME}_{secrets.token_hex(2)}'

            ## Connect to LTMS
            resp = self.connect_srv(sid, arrival_time.isoformat())
            assert not resp.transit_time < 0, resp.transit_time

            transit_time = resp.transit_time
            latest_reserve_time = datetime.fromisoformat(resp.latest_reserve_time)

            departure_time = arrival_time + timedelta(seconds=transit_time)

            ## Set up session
            self.sessions_add(sid,
                              sid=sid,
                              lock=debuggable_lock(f'{sid}_lock', RLock()),
                              parent=parent,
                              entry_loc=entry_loc,
                              exit_loc=exit_loc,
                              limits=None,
                              arrival_time=arrival_time,
                              latest_reserve_time=latest_reserve_time,
                              departure_time=departure_time)

            parent_sid = (None if parent is None else parent['sid']) 
            rospy.loginfo('\n'.join([
                'New session created!',
                f'  Name:                   {sid}',
                f'  Parent ID:              {parent_sid}',
                f'  entry_loc:              {entry_loc}',
                f'  exit_loc:               {exit_loc}',
                f'  arrival_time:           {arrival_time}',
                f'  departure_time:         {departure_time}',
                f'  latest_reserve_time:    {latest_reserve_time}',
                f'  Time now:               {datetime.utcnow()}',
            ]))

            return sid

    def sessions_update(self):
        now = datetime.utcnow()

        assert self.active_sid is not None, 'We always assume that when sessions_update is run, there is some prev session'
        assert self.session_order, 'We always assume that when sessions_update is run, there is some prev session'

        iter_opts = dict(order=True,
                         lock_all=False,
                         skip=self.reserved | {self.reserving_sid},
                         name='sessions_update [Notify]')

        i = 0
        for i, (sid, sess) in enumerate(self.iter_sessions(**iter_opts)):
            # Update un-reserved sessions. Reserved sessions are static.
            if sess['latest_reserve_time'] < now:
                rospy.logdebug(f'Want to notify {sid}, but after latest_reserve_time.')
                continue

            arrival_time = sess['parent']['departure_time']

            rospy.logdebug('\n'.join([
                'Notify!',
                f'  Name:                   {sid}',
                f'  Notified arrival_time:  {arrival_time}',
                f'  Time now:               {datetime.utcnow()}',
            ]))

            resp = self.notify_srv(sid, arrival_time.isoformat())
            assert resp.transit_time > 0, f'Notification error for {sid}: {resp.transit_time}'
            sess['latest_reserve_time'] = datetime.fromisoformat(resp.latest_reserve_time)
            sess['arrival_time'] = sess['parent']['departure_time']
            sess['departure_time'] = sess['arrival_time'] + timedelta(seconds=resp.transit_time)

        for j in range(i+1, self.NUM_SESSIONS_AHEAD):
            # Append new sessions
            parent_sid = sid if j>1 else self.active_sid # if j>1 then sid comes from above for-loop
            sid = self.create_session_after_parent(parent_sid)
            self.session_order.append(sid)

    def reserve(self, sid):
        for sess in self.select_session(sid, strict=True, lock_all=False, name='reserve'):

            rospy.logdebug('Reserving for %s', sid)
            start_time = datetime.utcnow()

            if sid in self.reserved | {self.reserving_sid}:
                rospy.logdebug(f'Already reserved/reserving for {sid}!')
                return True # success

            resp = self.reserve_srv(name=sid,
                                    entry=sess['entry_loc'],
                                    exit=sess['exit_loc'],
                                    time_ref=sess['arrival_time'].isoformat(),
                                    earliest_entry=0,
                                    latest_entry=3) # Let server choose when exactly to enter
        
            if not resp.success:
                rospy.loginfo('Reservation for %s failed: %s', sid, resp.reason)
                return False # success
            
            rospy.logdebug(f'Reservation succeeded! Took: {datetime.utcnow() - start_time}')

            sess['time_ref'] = datetime.fromisoformat(resp.time_ref)
            sess['earliest_entry'] = resp.earliest_entry
            sess['latest_entry'] = resp.latest_entry
            sess['earliest_exit'] = resp.earliest_exit
            sess['latest_exit'] = resp.latest_exit
        
            sess['arrival_time'] = sess['time_ref'] 
            sess['arrival_time'] += timedelta(seconds=(sess['earliest_entry'] + sess['latest_entry'])/2)
            sess['departure_time'] = sess['time_ref']
            sess['departure_time'] += timedelta(seconds=(sess['earliest_exit'] + sess['latest_exit'])/2)

            rospy.logdebug('Reservation complete for %s', sid)
            rospy.logdebug('\n'.join([
                'Reservation completed!',
                f'  Name:           {sid}',
                f'  Request Time:   {start_time}',
                f'  Time now:       {datetime.utcnow()}',
            ]))
            return True # success

    def block_until(self, cond=None, time=None, sleep_time=0.05):
        if cond is None and time is not None:
            cond = lambda: time <= datetime.utcnow()
        assert cond is not None, 'Must supply release condition'
        while not rospy.is_shutdown():
            if cond(): break
            rospy.sleep(sleep_time)

    def switch_session(self):
        with self.sessions_lock('switch_session'):
            self.active_sid, *self.session_order = self.session_order
            rospy.loginfo('Switching to session ID: %s', self.active_sid)
            assert self.active_sid in self.reserved, 'Switching to unreserved session'

    def run(self):

        with self.sessions_lock('run [Initialization]'):

            ## Initialize sessions
            arrival_time = datetime.utcnow() + self.INIT_WAIT
            entry_loc = 'init'
            exit_loc = self.choose_exit(entry_loc)
            init_sid = self.create_session(arrival_time, entry_loc, exit_loc)
            self.session_order.append(init_sid)
            rospy.loginfo('Initial session ID: %s', init_sid)

            # append following sessions
            parent_sid = init_sid
            for i in range(self.NUM_SESSIONS_AHEAD):
                sid = self.create_session_after_parent(parent_sid)
                self.session_order.append(sid)
                parent_sid = sid

        ## Wait for initial session to be reserved
        rospy.loginfo('Waiting for initial session to be reserved...')
        self.block_until(lambda: init_sid in self.reserved)

        ## Wait for start time

        rospy.loginfo('Waiting to reach start time...')

        init_arrival_time = self.read_session_prop(init_sid, 'arrival_time') # might have changed since reservation
        assert datetime.utcnow() < init_arrival_time, 'You probably want to increase INIT_WAIT'
        self.block_until(time=init_arrival_time)

        ## Set active sid and startbackground session updater 

        self.switch_session()

        rospy.Timer(rospy.Duration(5), lambda event: self.sessions_update())

        ## Start main loop

        rospy.loginfo('Starting main loop...')

        steering, velocity = 0, 0

        while not rospy.is_shutdown():

            with self.sessions_lock('run [Get Limits]'):

                active_sess = self.sessions[self.active_sid]
                assert self.active_sid in self.reserved
                print(active_sess)

                switch_time = (active_sess['time_ref']
                               + timedelta(seconds=active_sess['earliest_exit']))
                
                if switch_time <= datetime.utcnow():
                    self.switch_session()
                    active_sess = self.sessions[self.active_sid]
            
                limits_mask = active_sess['limits']

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

