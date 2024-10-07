#! /usr/bin/env python3.9

# Plain python imports
import numpy as np
import secrets
import json
from threading import RLock
from math import pi, ceil, floor
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
from contextlib import contextmanager

import rospy
from ltms.srv import Connect, Notify, Reserve
from svea_msgs.msg import State as StateMsg

import jax.numpy as jnp

import hj_reachability as hj
import hj_reachability.shapes as shp
from ltms_util import Solver, create_chaos

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


def service_wrp(srv_cls, method=False):
    Req  = srv_cls._request_class
    Resp = srv_cls._response_class
    cond = lambda args, kwds: (not kwds 
                               and len(args) == 1 
                               and isinstance(args[0], Req))
    def decorator(f):
        if method:
            @wraps(f)
            def wrapper(self, *args, **kwds):
                req = (args[0] if cond(args, kwds) else Req(*args, **kwds))
                resp = Resp()
                f(self, req, resp)
                return resp
        else:
            @wraps(f)
            def wrapper(*args, **kwds):
                req = (args[0] if cond(args, kwds) else Req(*args, **kwds))
                resp = Resp()
                resp = Resp()
                f(req, resp)
                return resp
        return wrapper
    return decorator

def around(l, e, n):
    if e not in l: return []
    i = l.index(e)
    N = len(l)
    return [l[(i+j) % N] for j in range(-n, n+1)]

class Server:

    AVOID_MARGIN = 0.4
    TIME_HORIZON = 5
    TIME_STEP = 0.2

    MAX_WINDOW_ENTRY = 2

    SESSION_TIMEOUT = timedelta(minutes=2)
    TRANSIT_TIME = 5 # [s] made up, roughly accurate
    COMPUTE_TIME = 3 # [s] made up, roughly accurate

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

    ENTRY_LOCATIONS = LOCATIONS + ['init']
    EXIT_LOCATIONS = LOCATIONS
    LOCATIONS += ['outside']
    PERMITTED_ROUTES.update({
        ('init', _exit): ('outside',)
        for _exit in 'center_ne center_ene center_e'.split()
    })

    def __init__(self):

        ## Initialize node

        rospy.init_node(self.__class__.__name__)

        ## Load parameters

        self.NAME = load_param('~name')

        self.DATA_DIR = load_param('~data_dir')
        self.DATA_DIR = Path(self.DATA_DIR)

        self.MODEL = load_param('~model', 'Bicycle4D')
        self.MODEL = vars(hj.systems)[self.MODEL]
        
        self.MIN_BOUNDS = load_param('~min_bounds', [-1.5, -1.5, -np.pi, +0.0])
        self.MIN_BOUNDS = np.array(self.MIN_BOUNDS)
        
        self.MAX_BOUNDS = load_param('~max_bounds', [+1.5, +1.5, +np.pi, +0.6])
        self.MAX_BOUNDS = np.array(self.MAX_BOUNDS)

        self.GRID_SHAPE = load_param('~grid_shape', [31, 31, 25, 7])
        self.GRID_SHAPE = tuple(self.GRID_SHAPE)

        # self.min_bounds = np.array([-1.5, -1.5, -np.pi, -np.pi/5, +0.0])
        # self.max_bounds = np.array([+1.5, +1.5, +np.pi, +np.pi/5, +0.6])
        # self.grid_shape = (31, 31, 25, 5, 7)
        # self.model = hj.systems.Bicycle5D

        ## Create simulators, models, managers, etc.

        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(self.MIN_BOUNDS, self.MAX_BOUNDS),
                                                                            self.GRID_SHAPE, periodic_dims=2)

        self.solver = Solver(grid=self.grid, 
                             time_step=self.TIME_STEP,
                             time_horizon=self.TIME_HORIZON,
                             accuracy='medium',
                             dynamics=dict(cls=self.model,
                                           min_steer=-pi * 5/4, 
                                           max_steer=+pi * 5/4,
                                           min_accel=-0.5, 
                                           max_accel=+0.5),
                             interactive=False)

        self.environment = self.load_environment()
        self.offline_passes = self.load_offline_analyses()

        self.sessions_lock = RLock()
        self.sessions = {}

        def clean_reservations_tmr(event):
            now = datetime.now().replace(microsecond=0)
            for id, reservation in self.get_reservations():
                if reservation['time_ref'] + timedelta(seconds=self.TIME_HORIZON) < now:
                    self.reservations.pop(id, None)
        rospy.Timer(rospy.Duration(1), clean_reservations_tmr)

        ## Advertise services

        rospy.Service('~connect', Connect, )
        rospy.Service('~reserve', ReserveCorridor, self.reserve_srv)

        ## Node initialized

        rospy.loginfo(f'{self.__class__.__name__} initialized!')

    def load_environment(self):
        out = {}
        for loc in self.LOCATIONS:
            filename = self.DATA_DIR / f'G{self.solver.code_grid}-{loc}.npy'
            if rospy.is_shutdown():
                break
            elif filename.exists():
                out[loc] = np.load(filename, allow_pickle=True)
                print(f'Loading {filename}')
            else:
                out.update(create_4way(self.grid, loc))
                print(f'Saving {filename}')
                np.save(filename, out[loc], allow_pickle=True)
        print('Environment done.')
        return out

    def load_offline_analyses(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        out = {}
        for (entry, exit), locs in self.PERMITTED_ROUTES.items():
            code = (f'G{self.solver.code_grid}'
                    f'D{self.solver.code_dynamics}'
                    f'T{self.solver.code_time}')
            filename = self.DATA_DIR / f'{code}-pass1-{entry}-{exit}.npy'
            if rospy.is_shutdown():
                break
            elif filename.exists():
                print(f'Loading {filename}')
                out[entry, exit] = np.load(filename, allow_pickle=True)
            else:
                constraints = shp.union(*[self.environment[loc] for loc in locs])

                output = self.solver.run_analysis('pass1',
                                                  exit=self.environment[exit],
                                                  constraints=constraints)
                
                print(f'Saving {filename}')
                np.save(filename, output['pass1'], allow_pickle=True)
                out[entry, exit] = output['pass1']
        
        print('Offline analyses done.')
        return out
    
    @property
    def iter_sessions(self):
        with self.sessions_lock:
            yield from self.sessions.items()

    @property
    def get_sessions(self):
        return list(self.iter_sessions)
    
    def sessions_add(self, key, dict1=None, **kwds):
        assert bool(dict1 is not None) ^ bool(kwds), 'Use either dict1 or keywords'
        with self.sessions_lock:
            self.sessions[key] = kwds if dict1 is None else dict1

    @property
    def get_reservations(self):
        return [(id_, sess['reservation']) for id_, sess in self.iter_sessions]
    
    def add_resevation(self, key, dict1=None, **kwds):
        assert bool(dict1 is not None) ^ bool(kwds), 'Use either dict1 or keywords'
        with self.sessions_lock:
            with self.sessions[key]['reservation_lock']:
                self.sessions[key]['reservation'] = kwds if dict1 is None else dict1
    
    @service_wrp(Connect, method=True)
    def connect_srv(self, req, resp):

        usr_id = req.usr_id
        its_id = f'its_{secrets.token_hex(4)}'

        @service_wrp(Notify)
        def notify_srv(req, resp):
            now = datetime.now().replace(microsecond=0)
            arrival_time = datetime.fromisoformat(req.arrival_time)

            num_unreserved = 0
            for id_, sess in self.get_sessions:
                if id_ == usr_id: continue
                if not 0 <= arrival_time - sess['arrival_time'] <= self.TIME_HORIZON: continue
                if not sess['reservation']:
                    num_unreserved += 1
            
            time_left = (arrival_time - now).total_seconds()
            time_needed = self.COMPUTE_TIME * (num_unreserved+1)
            if time_left <= time_needed:
                # will be negative; the time we're late with. minus sign indicates an error. 
                resp.transit_time = time_left - time_needed
                return

            self.sessions[usr_id]['arrival_time'] = arrival_time
            self.sessions[usr_id]['session_timeout'] = arrival_time + self.SESSION_TIMEOUT
            resp.transit_time = self.TRANSIT_TIME
        
        def state_cb(state_msg):
            now = datetime.now().replace(microsecond=0)

            x = state_msg.x
            y = state_msg.y
            h = state_msg.yaw
            v = state_msg.velocity

            time_ref = self.sessions[usr_id]['reservation']['time_ref']
            pass4 = self.sessions[usr_id]['reservation']['analysis']['pass4']
            state = np.array([x, y, h, 0, v])
            i = (now - time_ref).total_seconds() // self.TIME_STEP

            if not 0 <= i < len(self.solver.timeline):
                return # outside timeline
            
            idx = (np.array([x, y, h]) - self.grid.domain.lo[:3]) / np.array(self.grid.spacings)[:3]
            idx = np.where(self.grid._is_periodic_dim, idx % np.array(self.grid.shape), idx)
            idx = np.round(idx).astype(int)

            if (idx < 0).any() or (self.grid.shape[:3] <= idx).any():
                return # outside grid

            mask, ctrl_vecs = self.solver.lrcs(pass4, state, i)

            limits_msg = mask.tobytes()
            self.sessions[usr_id]['DrivingLimits'].publish(limits_msg)

        @service_wrp(Reserve)
        def reserve_srv(req, resp):
            resp.success = False
            resp.reason = 'Unknown.'

            if req.entry not in self.ENTRY_LOCATIONS:
                resp.reason = f"Illegal entry region: '{req.entry}'."
                return
            if req.exit not in self.EXIT_LOCATIONS:
                resp.reason = f"Illegal exit region: '{req.exit}'."
                return
            if (req.entry, req.exit) not in self.PERMITTED_ROUTES:
                resp.reason = f"Illegal route through region: '{req.entry}' -> '{req.exit}'."
                return
            
            try:
                time_ref = datetime.fromisoformat(req.time_ref)
            except Exception:
                resp.reason = f"Malformed ISO time: '{req.time_ref}'."
                return
            
            notify_resp = notify_srv(arrival_time=req.time_ref)
            if notify_resp.transit_time < 0: # catch notify error
                resp.reason = f'Reservation late: {-notify_resp.transit_time} s too late'
                return

            rospy.loginfo(f"Reservation request from '{req.name}': {req.entry} -> {req.exit}")

            with self.reservations_lock:
                self.reserve(usr_id, req, resp)

            if not resp.success:
                rospy.loginfo(f"Reservation {resp.id} ({req.name}) recjected: {resp.reason}")
                return

            rospy.loginfo(f"Reservation {resp.id} ({req.name}) approved.")
            
            with self.sessions_lock:
                self.sessions[usr_id]['DrivingLimits'] = \
                    rospy.Publisher(f'/connz/{its_id}/limits', bytes)
                
                self.sessions[usr_id]['UserState'] = \
                    rospy.Subscriber(f'/connz/{usr_id}/state', StateMsg, ...)

        arrival_time = datetime.fromisoformat(req.arrival_time)
        session_timeout = arrival_time + self.SESSION_TIMEOUT

        self.sessions_add(usr_id,
                          Notify=rospy.Service(f'/connz/{its_id}/notify', Notify, notify_srv),
                          Reserve=rospy.Service(f'/connz/{its_id}/reserve', Reserve, reserve_srv),
                          self=its_id, user=usr_id,
                          arrival_time=arrival_time,
                          session_timeout=session_timeout,
                          reservation={},
                          reservation_lock=RLock())

        resp.its_id = its_id

    def resolve_dangers(self, time_ref):

        td_horizon = timedelta(seconds=self.TIME_HORIZON)
        
        dangers = []
        for _, reservation in self.get_reservations():
            earliest_overlap = max(time_ref, reservation['time_ref'])
            latest_overlap = min(time_ref + td_horizon, reservation['time_ref'] + td_horizon)
            overlap = (latest_overlap - earliest_overlap).total_seconds()

            if not 0 < overlap:
                continue
            
            danger = np.ones(self.solver.timeline.shape + self.grid.shape)
            if time_ref < earliest_overlap:
                # HPV:     [-----j----)
                # LPV: [---i-----)
                i_offset = (earliest_overlap - time_ref).total_seconds()
                j_offset = (latest_overlap - reservation['time_ref']).total_seconds()
                i = ceil(i_offset / self.TIME_STEP)
                j = ceil(j_offset / self.TIME_STEP)
                danger[i:] = reservation['analysis']['pass4'][:j]
            else: 
                # HPV: [---i-----)
                # LPV:     [-----j----)
                i_offset = (earliest_overlap - reservation['time_ref']).total_seconds()
                j_offset = (latest_overlap - time_ref).total_seconds()
                i = ceil(i_offset / self.TIME_STEP)
                j = ceil(j_offset / self.TIME_STEP)
                danger[:j] = reservation['analysis']['pass4'][i:i+j]
            dangers.append(danger)

        if not dangers:
            rospy.loginfo('Intersection free!')

        return dangers

    def reserve(self, usr_id, req, resp):
        try:
            earliest_entry = round(max(req.earliest_entry, 0), 1)
            latest_entry = round(min(req.latest_entry, self.TIME_HORIZON), 1)
            
            offset = round(floor(earliest_entry) + (earliest_entry % self.TIME_STEP), 1)
            time_ref += timedelta(seconds=offset)
            earliest_entry -= offset
            latest_entry -= offset
            assert 0 <= earliest_entry <= latest_entry <= self.TIME_HORIZON, \
                f'Negotiation Failed: Invalid window offsetting (offset={offset})'

            max_window_entry = round(min(latest_entry - earliest_entry, self.MAX_WINDOW_ENTRY), 1)
            assert max_window_entry, 'Negotiation Failed: Invalid entry window requested'
            
            dangers = self.resolve_dangers(time_ref)
            output = self.solver.run_analysis('pass4',
                                              max_window_entry=max_window_entry,
                                              pass1=self.offline_passes[req.entry, req.exit],
                                              entry=self.environment[req.entry],
                                              exit=self.environment[req.exit],
                                              dangers=dangers)
            
            earliest_entry = max(earliest_entry, output['earliest_entry'])
            latest_entry = min(latest_entry, output['latest_entry'])
            earliest_exit = output['earliest_exit']
            latest_exit = output['latest_exit']
            assert 0 < latest_entry - earliest_entry, 'Negotiation Faild: No time window to enter region'

        except AssertionError as e:
            msg, = e.args
            resp.reason = f'Reservation Error: {msg}'
            return

        else:
            self.add_resevation(usr_id,
                                name=req.name, 
                                time_ref=time_ref,
                                entry=req.entry, exit=req.exit,
                                earliest_entry=earliest_entry,
                                latest_entry=latest_entry,
                                earliest_exit=earliest_exit,
                                latest_exit=latest_exit,
                                analysis=output)
            
            flat_corridor = shp.project_onto(output['pass4'], 1, 2) <= 0

            resp.time_ref = time_ref.isoformat()
            resp.earliest_entry = earliest_entry
            resp.latest_entry = latest_entry
            resp.earliest_exit = earliest_exit
            resp.latest_exit = latest_exit
            resp.shape = list(flat_corridor.shape)
            resp.corridor = flat_corridor.tobytes()
            resp.success = True
            resp.reason = ''

            # np.save(f'/svea_ws/src/ltms/data/{req.name}.npy', corridor, allow_pickle=True)
            
            # meta['time_ref'] = time_ref.isoformat()
            # with open(f'/svea_ws/src/ltms/data/{req.name}.json', 'w') as f:
            #     json.dump(meta, f)

    def run(self):
        rospy.spin()

if __name__ == '__main__':

    ## Start node ##
    Server().run()

