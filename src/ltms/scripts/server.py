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

import rospy
from ltms.msg import NamedBytes
from ltms.srv import Connect, Notify, Reserve
from svea_msgs.msg import VehicleState as StateMsg

import hj_reachability as hj
import hj_reachability.shapes as shp
from ltms_util import Solver, create_chaos
from nats_ros_connector.nats_manager import NATSManager

from contextlib import contextmanager

from std_msgs.msg import String

def debuggable_lock(name, lock):
    @contextmanager
    def ctx(caller):
        rospy.logdebug('%s: %s acquiring', caller, name)
        with lock:
            rospy.logdebug('%s: %s acquired', caller, name)
            yield
        rospy.logdebug('%s: %s released', caller, name)
    return ctx

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

_LOCATIONS = ['left', 'top', 'right', 'bottom']
_PERMITTED_ROUTES = {
    (_entry, _exit): ('full',)
    for _entry in _LOCATIONS
    for _exit in set(_LOCATIONS) - {_entry}
}

class Server:

    AVOID_MARGIN = 0.4
    TIME_HORIZON = 15
    TIME_STEP = 0.2

    MAX_WINDOW_ENTRY = 2

    SESSION_TIMEOUT = timedelta(seconds=30)
    TRANSIT_TIME = 15 # [s] made up, roughly accurate
    COMPUTE_TIME = 10 # [s] made up, roughly accurate

    ENTRY_LOCATIONS = _LOCATIONS + ['init']
    EXIT_LOCATIONS = _LOCATIONS
    LOCATIONS = _LOCATIONS + ['full', 'init']
    PERMITTED_ROUTES = _PERMITTED_ROUTES | {
        ('init', _exit): ('full',)
        for _exit in 'left'.split()
    }

    def __init__(self):

        ## Initialize node

        rospy.init_node(self.__class__.__name__, log_level=rospy.INFO)

        ## Load parameters

        self.NAME = load_param('~name')

        self.DATA_DIR = load_param('~data_dir')
        self.DATA_DIR = Path(self.DATA_DIR)

        self.MODEL = load_param('~model', 'Bicycle4D')
        self.MODEL = vars(hj.systems)[self.MODEL]

        self.GRID_SHAPE = load_param('~grid_shape', [31, 31, 25, 7])
        self.GRID_SHAPE = tuple(map(int, self.GRID_SHAPE))

        self.MIN_BOUNDS = load_param('~min_bounds', [-1.5, -1.5, -np.pi, +0.0])
        self.MIN_BOUNDS = np.array([eval(x) if isinstance(x, str) else x for x in self.MIN_BOUNDS])
        
        self.MAX_BOUNDS = load_param('~max_bounds', [+1.5, +1.5, +np.pi, +0.6])
        self.MAX_BOUNDS = np.array([eval(x) if isinstance(x, str) else x for x in self.MAX_BOUNDS])

        self.nats_mgr = NATSManager()

        ## Create simulators, models, managers, etc.

        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(self.MIN_BOUNDS, self.MAX_BOUNDS),
                                                                            self.GRID_SHAPE, periodic_dims=2)

        self.solver = Solver(grid=self.grid, 
                             time_step=self.TIME_STEP,
                             time_horizon=self.TIME_HORIZON,
                             accuracy='low',
                             dynamics=dict(cls=self.MODEL,
                                           min_steer=-pi * 5/4, 
                                           max_steer=+pi * 5/4,
                                           min_accel=-0.5, 
                                           max_accel=+0.5),
                             interactive=False)

        self.environment = self.load_environment()
        self.offline_passes = self.load_offline_analyses()

        self.sessions_lock = debuggable_lock('sessions_lock', RLock())
        self.sessions = {}

        def clean_sessions_tmr(event):
            now = datetime.now().replace(microsecond=0)
            with self.sessions_lock('clean_sessions_tmr'):
                for sid, sess in self.get_sessions:
                    if sess['session_timeout'] < now:
                        del self.sessions[sid]
        rospy.Timer(rospy.Duration(2), clean_sessions_tmr)

        ## Advertise services

        self.Connect = self.nats_mgr.new_service('/server/connect', Connect, self.connect_srv)
        self.Notify = self.nats_mgr.new_service('/server/notify', Notify, self.notify_srv)
        self.Resere = self.nats_mgr.new_service('/server/reserve', Reserve, self.reserve_srv)
        
        self.Limits = self.nats_mgr.new_publisher(f'/server/limits', NamedBytes, queue_size=5)
        self.State = self.nats_mgr.new_subscriber(f'/server/state', StateMsg, self.state_cb)

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
                out.update(create_chaos(self.grid, loc))
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
        with self.sessions_lock('iter_sessions'):
            yield from self.sessions.items()

    @property
    def get_sessions(self):
        with self.sessions_lock('get_sessions'):
            return list(self.sessions.items())
    
    def sessions_add(self, sid, dict1=None, **kwds):
        assert bool(dict1 is not None) ^ bool(kwds), 'Use either dict1 or keywords'
        with self.sessions_lock('sessions_add'):
            self.sessions[sid] = kwds if dict1 is None else dict1

    def select_session(self, sid):
        with self.sessions_lock('select_session'):
            if sid in self.sessions:
                yield self.sessions[sid]

    @property
    def get_reservations(self):
        return [(sid, sess['reservation']) for sid, sess in self.iter_sessions if sess['reserved']]
    
    def add_reservation(self, sid, dict1=None, **kwds):
        assert bool(dict1 is not None) ^ bool(kwds), 'Use either dict1 or keywords'
        with self.sessions_lock('add_reservation'):
            for sess in self.select_session(sid):
                sess['reservation'] = kwds if dict1 is None else dict1
                sess['reserved'] = True
    
    @service_wrp(Connect, method=True)
    def connect_srv(self, req, resp):

        usr_id = req.usr_id
        its_id = f'its_{secrets.token_hex(4)}'

        rospy.logdebug(f'> {req.usr_id} will get here at {req.arrival_time}')

        arrival_time = datetime.fromisoformat(req.arrival_time)
        session_timeout = arrival_time + self.SESSION_TIMEOUT

        rospy.logdebug(f'>> timeout at {session_timeout}')

        self.sessions_add(usr_id,
                          arrival_time=arrival_time,
                          session_timeout=session_timeout,
                          reserved=False,
                          reservation={})

        resp.its_id = its_id
        resp.transit_time = self.TRANSIT_TIME

        rospy.logdebug(f'>> session added, {resp.its_id=}')

    @service_wrp(Notify, method=True)
    def notify_srv(self, req, resp):
        now = datetime.now().replace(microsecond=0)
        arrival_time = datetime.fromisoformat(req.arrival_time)

        resp.transit_time = self.TRANSIT_TIME

        for sess in self.select_session(req.usr_id):
            if sess['reserved']: return # don't update
            sess['arrival_time'] = arrival_time
            sess['session_timeout'] = arrival_time + self.SESSION_TIMEOUT

        num_unreserved = 0
        for sid, sess in self.get_sessions:
            if sid == req.usr_id: continue
            if not 0 <= (arrival_time - sess['arrival_time']).total_seconds() <= self.TIME_HORIZON: continue
            if not sess['reserved']:
                num_unreserved += 1

        time_left = (arrival_time - now).total_seconds()
        time_needed = self.COMPUTE_TIME * (num_unreserved+1)
        if time_left <= time_needed:
            # will be negative; the time we're late with. minus sign indicates an error. 
            resp.transit_time = time_left - time_needed

    def state_cb(self, state_msg):
        now = datetime.now().replace(microsecond=0)

        usr_id = state_msg.child_frame_id
        x = state_msg.x
        y = state_msg.y
        h = state_msg.yaw
        v = state_msg.v

        if usr_id not in self.sessions:
            return # not connected yet

        for sess in self.select_session(usr_id):
            if not sess['reserved']: return # not reserved so not limits avail
            time_ref = sess['reservation']['time_ref']
            pass4 = sess['reservation']['analysis']['pass4']    
            break
        else:
            return # not connected
        

        state = np.array([x, y, h, 0, v])
        i = (now - time_ref).total_seconds() // self.TIME_STEP

        if not 0 <= i < len(self.solver.timeline):
            rospy.loginfo('State outside of timeline for Limits: %s', usr_id)
            return # outside timeline
        
        idx = (np.array([x, y, h]) - self.grid.domain.lo[:3]) / np.array(self.grid.spacings)[:3]
        idx = np.where(self.grid._is_periodic_dim[:3], idx % np.array(self.grid.shape[:3]), idx)
        idx = np.round(idx).astype(int)

        if (idx < 0).any() or (self.grid.shape[:3] <= idx).any():
            rospy.loginfo('State outside of grid for Limits: %s', usr_id)
            return # outside grid

        mask, ctrl_vecs = self.solver.lrcs(pass4, state, ceil(i))

        limits_msg = NamedBytes(usr_id, mask.tobytes())
        self.Limits.publish(limits_msg)
        rospy.loginfo('Sending Limits')

    @service_wrp(Reserve, method=True)
    def reserve_srv(self, req, resp):
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
        
        notify_resp = self.notify_srv(arrival_time=req.time_ref)
        if notify_resp.transit_time < 0: # catch notify error
            resp.reason = f'Reservation late: {-notify_resp.transit_time} s too late'
            return

        rospy.loginfo(f"Reservation request from '{req.name}': {req.entry} -> {req.exit}")

        self.reserve(req.name, req, resp)

        if not resp.success:
            rospy.loginfo(f"Reservation for {req.name} recjected: {resp.reason}")
            return

        rospy.loginfo(f"Reservation for {req.name} approved.")

    def reserve(self, sid, req, resp):

        try:
            time_ref = datetime.fromisoformat(req.time_ref)
        except Exception:
            resp.reason = f"Malformed ISO time: '{req.time_ref}'."
            return
        
        # Debugging path
        save_path = None # to disable
        save_path = Path(f'/svea_ws/src/ltms/data/{sid}')
        save_path.mkdir(exist_ok=True)

        result = {}
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
            dangers = [danger
                       for _id, danger in dangers.items()
                       if _id[:5] != sid[:5]]

            self.solver.run_analysis('pass2', 'pass3', 'pass4',
                                     min_window_entry=1,  max_window_entry=max_window_entry,
                                     min_window_exit=1, max_window_exit=2.0,
                                     pass1=self.offline_passes[req.entry, req.exit],
                                     entry=self.environment[req.entry],
                                     exit=self.environment[req.exit],
                                     dangers=dangers,
                                     result=result,
                                     save_path=save_path,
                                     interactive=True)

            earliest_entry = max(earliest_entry, result['earliest_entry'])
            latest_entry = min(latest_entry, result['latest_entry'])
            earliest_exit = result['earliest_exit']
            latest_exit = result['latest_exit']
            assert 0 < latest_entry - earliest_entry, 'Negotiation Faild: No time window to enter region'

        except AssertionError as e:
            msg, = e.args
            resp.reason = f'Reservation Error: {msg}'

        else:
            self.add_reservation(sid,
                                 name=req.name, 
                                 time_ref=time_ref,
                                 entry=req.entry, exit=req.exit,
                                 earliest_entry=earliest_entry,
                                 latest_entry=latest_entry,
                                 earliest_exit=earliest_exit,
                                 latest_exit=latest_exit,
                                 analysis=result)
            
            # flat_corridor = shp.project_onto(result['pass4'], 1, 2) <= 0

            resp.time_ref = time_ref.isoformat()
            resp.earliest_entry = earliest_entry
            resp.latest_entry = latest_entry
            resp.earliest_exit = earliest_exit
            resp.latest_exit = latest_exit
            # resp.shape = list(flat_corridor.shape)
            # resp.corridor = flat_corridor.tobytes()
            resp.success = True
            resp.reason = ''

        # with open(f'/svea_ws/src/ltms/data/{sid}.json', 'w') as f:
        #     json.dump({
        #         'time_ref': time_ref.isoformat(),
        #         'earliest_entry': earliest_entry,
        #         'latest_entry': latest_entry,
        #         'earliest_exit': earliest_exit,
        #         'latest_exit': latest_exit,
        #         'output_earliest_entry': result['earliest_entry'],
        #         'output_latest_entry': result['latest_entry'],
        #         'output_earliest_exit': result['earliest_exit'],
        #         'output_latest_exit': result['latest_exit'],
        #     }, f)
        
    def resolve_dangers(self, time_ref, quiet=False):

        td_horizon = timedelta(seconds=self.TIME_HORIZON)
        
        dangers = {}
        for sid, reservation in self.get_reservations:
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
            dangers[sid] = danger

        if not dangers and not quiet:
            rospy.loginfo('Intersection free!')

        return dangers

    def tube_to_marker(self, tube):
        pass

    def run(self):
        
        rate = rospy.Rate(5)

        while not rospy.is_shutdown():
            now = datetime.now()

            for tube in self.resolve_dangers(now, quiet=True):
                marker = self.tube_to_marker(tube)

            rate.sleep()

if __name__ == '__main__':

    ## Start node ##
    Server().run()

