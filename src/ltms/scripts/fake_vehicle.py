#! /usr/bin/env python3

import numpy as np
import secrets
from datetime import datetime, timedelta

# SVEA imports
from svea.interfaces import ActuationInterface
from svea_msgs.msg import VehicleState as VehicleStateMsg

from session_mgr import *

# ROS imports
import rospy
from ltms.msg import NamedBytes
from ltms.srv import Connect, Notify, Reserve
from ltms_util.debuggable_lock import *
import message_filters as mf
from geometry_msgs.msg import PoseStamped, TwistStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from threading import RLock

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

        self.sessions = SessionMgr()

        rospy.Timer(rospy.Duration(5),
                    lambda event: self.updater())

        rospy.Timer(rospy.Duration(2), 
                    lambda event: self.reserver())

        ## Rate
        
        self.rate = rospy.Rate(10)

        state = VehicleStateMsg()
        state.child_frame_id = self.NAME
        state.x = +1.75
        state.y = -1.75
        state.yaw = np.pi
        state.v = 0
        self.state = state

        ## Initiatlize interfaces

        self.actuator = ActuationInterface(self.NAME).start()

        ## Create service proxies
        self.connect_srv = rospy.ServiceProxy('/server/connect', Connect)
        self.notify_srv = rospy.ServiceProxy('/server/notify', Notify)
        self.reserve_srv = rospy.ServiceProxy('/server/reserve', Reserve)
        
        self.target = None
        def limits_cb(msg):
            opts = dict(lock_all=False,
                        _dbgname='limits_cb')
            for sess in self.sessions.select(msg.name, **opts):
                rospy.logdebug('\n'.join([
                    f'Received new driving limits.',
                    f'  name: {msg.name}',
                    f'  delta: {(rospy.Time.now() - msg.stamp).to_sec()} s',
                ]))
                limits = np.frombuffer(bytes([x+128 for x in msg.mask]), dtype=bool)
                sess['limits'] = limits.reshape(len(self.U1_VEC), len(self.U2_VEC))
        self.State = rospy.Publisher(f'/server/state', VehicleStateMsg, queue_size=5)
        self.Limits = rospy.Subscriber(f'/server/limits', NamedBytes, limits_cb)

        def limits_tmr(event):
            sid = self.sessions.active_sid
            if sid is not None:
                self.state.child_frame_id = sid
                self.State.publish(self.state)
        rospy.Timer(rospy.Duration(0.1), limits_tmr)

        ## Node initialized
        rospy.loginfo(f'{self.__class__.__name__} initialized!')
    
    def choose_exit(self, entry):
        permitted_exits = [exit for exit in self.EXIT_LOCATIONS if (entry, exit) in self.PERMITTED_ROUTES]
        return np.random.choice(permitted_exits)

    def create_session_after_parent(self, parent_sid):

        opts = dict(strict=True, 
                    _dbgname='create_session_after_parent')
        for parent_sess in self.sessions.select(parent_sid, **opts):
            arrival_time = parent_sess['departure_time']
            entry_loc = parent_sess['exit_loc']

        exit_loc = self.choose_exit(entry_loc)

        return self.create_session(arrival_time, entry_loc, exit_loc, parent=parent_sess)

    def create_session(self, arrival_time, entry_loc, exit_loc, parent=None):
        ## Create new random session id
        sid = f'{self.NAME}_{secrets.token_hex(2)}'

        ## Connect to LTMS
        resp = self.connect_srv(sid, arrival_time.isoformat())
        assert not resp.transit_time < 0, resp.transit_time

        transit_time = resp.transit_time
        latest_reserve_time = datetime.fromisoformat(resp.latest_reserve_time)

        departure_time = arrival_time + timedelta(seconds=transit_time)

        ## Set up session
        self.sessions.add(sid,
                          parent=parent,
                          entry_loc=entry_loc,
                          exit_loc=exit_loc,
                          limits=None,
                          arrival_time=arrival_time,
                          latest_reserve_time=latest_reserve_time,
                          departure_time=departure_time,
                          _addorder=True,
                          _dbgname=f'create_session [{sid}]')

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

    def updater(self):

        if self.sessions.active_sid is None: return

        ## Update un-reserved sessions. Reserved sessions are static.

        opts = dict(order=True,
                    lock_all=False,
                    skip=self.sessions._reserved | {self.sessions.reserving_sid},
                    _dbgname='sessions_update [Notify]')

        i = -1
        for i, (sid, sess) in enumerate(self.sessions.iterate(**opts)):
            now = datetime.utcnow()

            if sess['latest_reserve_time'] < now:
                rospy.logwarn(f'Want to update-notify {sid}, but after latest_reserve_time.')
                continue

            if sess['parent'] is None:
                rospy.logwarn(f'Want to update-notify {sid}, but missing parent. Cannot determine departure_time.')
                continue

            arrival_time = sess['parent']['departure_time']

            rospy.logdebug('\n'.join([
                'Notify!',
                f'  Name:                   {sid}',
                f'  Notified arrival_time:  {arrival_time}',
                f'  Time now:               {now}',
            ]))

            resp = self.notify_srv(sid, arrival_time.isoformat())
            assert resp.transit_time > 0, f'Notification error for {sid}: {resp.transit_time}'
            sess['latest_reserve_time'] = datetime.fromisoformat(resp.latest_reserve_time)
            sess['arrival_time'] = sess['parent']['departure_time']
            sess['departure_time'] = sess['arrival_time'] + timedelta(seconds=resp.transit_time)

        ## Append new sessions

        for i in range(i+1, self.NUM_SESSIONS_AHEAD):
            parent_sid = sid if i>0 else self.sessions.active_sid # if i>0 then sid comes from above for-loop
            sid = self.create_session_after_parent(parent_sid)

    def reserver(self):

        ## Find next to reserve

        opts = dict(order=True, 
                    lock_all=False,
                    skip=self.sessions._reserved, 
                    _dbgname='reserver')
        for sid, sess in self.sessions.iterate(**opts):
            if sess['latest_reserve_time'] - datetime.utcnow() <= self.RES_TIME_LIMIT:
                break
        else:
            return

        rospy.logdebug(f'>>> {self.sessions._order=}')

        ## Do the reservation

        mark1_time = datetime.utcnow()

        opts = dict(lock_all=False, _dbgname='reserve')
        for sess in self.sessions.select_to_reserve(sid, **opts):

            mark2_time = datetime.utcnow()

            success = False
            while not success:
                if rospy.is_shutdown(): return

                rospy.loginfo('Reserving for %s', sid)
                mark3_time = start_time = datetime.utcnow()

                if sess['reserved']:
                    rospy.logdebug(f'Already reserved for {sid}!')
                    success = True
                    break

                resp = self.reserve_srv(name=sid,
                                        entry=sess['entry_loc'],
                                        exit=sess['exit_loc'],
                                        time_ref=sess['arrival_time'].isoformat(),
                                        earliest_entry=0,
                                        latest_entry=3) # Let server choose when exactly to enter
                
                success = resp.success
                if not success:
                    rospy.logdebug('\n'.join([
                        'Reservation failed:',
                        f'  Name:   {sid}',
                        f'  Mark 1:     {mark1_time}',
                        f'  Mark 2:     {mark2_time}',
                        f'  Mark 3:     {mark3_time}',
                        f'  Reason: {resp.reason}',
                    ]))
                    rospy.signal_shutdown('Fatal Error')
                else: 
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

                    now = datetime.utcnow()
                    rospy.logdebug('Reservation complete for %s', sid)
                    rospy.logdebug('\n'.join([
                        'Reservation done:',
                        f'  Name:           {sid}',
                        f'  Request Time:   {start_time}',
                        f'  Time now:       {now}',
                        f'  Took:           {now - start_time}',
                    ]))

    def block_until(self, cond=None, time=None, sleep_time=0.05):
        if cond is None and time is not None:
            cond = lambda: time <= datetime.utcnow()
        assert cond is not None, 'Must supply release condition'
        while not rospy.is_shutdown():
            if cond(): break
            rospy.sleep(sleep_time)

    def run(self):

        ## Initialize first sessions
        arrival_time = datetime.utcnow() + self.INIT_WAIT
        entry_loc = 'init'
        exit_loc = self.choose_exit(entry_loc)
        init_sid = self.create_session(arrival_time, entry_loc, exit_loc)
        rospy.loginfo('Initial session ID: %s', init_sid)

        ## Wait for initial session to be reserved
        rospy.loginfo('Waiting for initial session to be reserved...')
        self.block_until(lambda: init_sid in self.sessions._reserved)

        ## Wait for start time

        rospy.loginfo('Waiting to reach start time...')

        init_arrival_time = self.sessions.read_prop(init_sid, 'arrival_time') # might have changed since reservation
        assert datetime.utcnow() < init_arrival_time, 'You probably want to increase INIT_WAIT'
        self.block_until(time=init_arrival_time)

        ## Set active sid and startbackground session updater 

        self.sessions.next()

        ## Start main loop

        rospy.loginfo('Starting main loop...')

        steering, velocity = 0, 0

        while not rospy.is_shutdown():

            opts = dict(_dbgname='run [Get Limits]')
            for active_sess in self.sessions.select_active(**opts):

                switch_time = (active_sess['time_ref']
                               + timedelta(seconds=active_sess['earliest_exit']))
                
                if switch_time <= datetime.utcnow():
                    self.sessions.next()
                    break
                
                limits_mask = active_sess['limits']

            else:

                if limits_mask is None:
                    rospy.logwarn('Missing driving limits')
                else:
                    rospy.sleep(0.001)

                rospy.loginfo(f'Steering: {steering}, Velocity: {velocity}')
                self.actuator.send_control(steering=steering, velocity=velocity)

                # Sleep
                self.rate.sleep()

if __name__ == '__main__':

    ## Start node ##
    vehicle = Vehicle()
    vehicle.run()

