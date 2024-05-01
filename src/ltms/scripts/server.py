#! /usr/bin/env python3.9

# Plain python imports
import numpy as np
import secrets
import json
from threading import RLock
from math import ceil, floor
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
from contextlib import contextmanager

import rospy
from ltms.srv import ReserveCorridor, SafeInput

import jax.numpy as jnp

import hj_reachability as hj
import hj_reachability.shapes as shp
from ltms_util import Solver, create_4way

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


def service(srv_cls):
    def decorator(f):
        @wraps(f)
        def wrapper(self, req):
            resp = srv_cls._response_class()
            f(self, req, resp)
            return resp
        return wrapper
    return decorator


class Server:

    AVOID_MARGIN = 0.4
    TIME_HORIZON = 5
    TIME_STEP = 0.2

    MAX_WINDOW_ENTRY = 2

    ENTRY_LOCATIONS = ['entry_s', 'entry_w', 'entry_n', 'entry_e']
    EXIT_LOCATIONS = ['exit_s', 'exit_w', 'exit_n', 'exit_e']
    LOCATIONS = [
        'center',
        'road_s', 'road_w', 'road_n', 'road_e',
        *ENTRY_LOCATIONS,
        *EXIT_LOCATIONS,
    ]

    PERMITTED_ROUTES = {
        # BIG OBS TO SELF: if entering from south then we're traveling in north direction
        # => => => 'entry_s' requires 'road_n'

        ('entry_s', 'exit_w'): ('road_n', 'road_w'),
        ('entry_s', 'exit_n'): ('road_n', 'road_n'),
        ('entry_s', 'exit_e'): ('road_n', 'road_e'),

        ('entry_w', 'exit_n'): ('road_e', 'road_n'),
        ('entry_w', 'exit_e'): ('road_e', 'road_e'),
        ('entry_w', 'exit_s'): ('road_e', 'road_s'),

        ('entry_n', 'exit_w'): ('road_s', 'road_w'),
        ('entry_n', 'exit_s'): ('road_s', 'road_s'),
        ('entry_n', 'exit_e'): ('road_s', 'road_e'),

        ('entry_e', 'exit_s'): ('road_w', 'road_s'),
        ('entry_e', 'exit_w'): ('road_w', 'road_w'),
        ('entry_e', 'exit_n'): ('road_w', 'road_n'),

        # U-turns
        ('entry_s', 'exit_s'): ('road_n', 'road_s'),
        ('entry_w', 'exit_w'): ('road_e', 'road_w'),
        ('entry_n', 'exit_n'): ('road_s', 'road_n'),
        ('entry_e', 'exit_e'): ('road_w', 'road_e'),
    }

    def __init__(self):
        """Init method for SocialNavigation class."""

        ## Initialize node

        rospy.init_node(self.__class__.__name__)

        ## Load parameters

        self.NAME = load_param('~name')
        self.DATA_DIR = load_param('~data_dir')

        self.DATA_DIR = Path(self.DATA_DIR)

        ## Create simulators, models, managers, etc.

        if False:
            self.min_bounds = np.array([-1.2, -1.2, -np.pi, -np.pi/5, +0.3])
            self.max_bounds = np.array([+1.2, +1.2, +np.pi, +np.pi/5, +0.8])
            self.grid_shape = (31, 31, 25, 7, 7)
            self.model = hj.systems.Bicycle5D
        else:
            self.min_bounds = np.array([-1.2, -1.2, -np.pi, +0.3])
            self.max_bounds = np.array([+1.2, +1.2, +np.pi, +0.8])
            self.grid_shape = (31, 31, 25, 7)
            self.model = hj.systems.Bicycle4D

        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(self.min_bounds, self.max_bounds),
                                                                            self.grid_shape,
                                                                            periodic_dims=2)

        self.solver = Solver(grid=self.grid, 
                             time_step=self.TIME_STEP,
                             time_horizon=self.TIME_HORIZON,
                             accuracy='medium',
                             dynamics=dict(cls=self.model,
                                           min_steer=-jnp.pi * 5/4, 
                                           max_steer=+jnp.pi * 5/4,
                                           min_accel=-0.5, 
                                           max_accel=+0.5),
                             interactive=False)

        self.environment = self.load_environment()
        self.offline_passes = self.load_offline_analyses()
        
        self.reservation_lock = RLock()
        self.reservations = {}

        def clean_reservations_tmr(event):
            now = datetime.now().replace(microsecond=0)
            for id, reservation in self.get_reservations():
                if reservation['time_ref'] + timedelta(seconds=self.TIME_HORIZON) < now:
                    self.reservations.pop(id, None)
        rospy.Timer(rospy.Duration(1), clean_reservations_tmr)

        ## Advertise services

        # rospy.Service('~save_input', SafeInput, self.safe_input_srv)
        rospy.Service('~request_corridor', ReserveCorridor, self.request_corridor_srv)

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
    
    def get_reservations(self, unzip=False):
        with self.reservation_lock:
            reservations = list(self.reservations.items())
            return zip(*reservations) if unzip else reservations
    
    def add_resevation(self, id, **data):
        with self.reservation_lock:
            self.reservations[id] = data

    @service(ReserveCorridor)
    def request_corridor_srv(self, req, resp):
        resp.success = False
        resp.reason = 'Unknown.'
        resp.id = secrets.token_hex(4)

        try:
            time_ref = datetime.fromisoformat(req.time_ref)
        except Exception:
            resp.reason = f"Malformed ISO time: '{req.time_ref}'."
            return
        
        if req.entry not in self.ENTRY_LOCATIONS:
            resp.reason = f"Illegal entry region: '{req.entry}'."
            return
        if req.exit not in self.EXIT_LOCATIONS:
            resp.reason = f"Illegal exit region: '{req.exit}'."
            return
        if (req.entry, req.exit) not in self.PERMITTED_ROUTES:
            resp.reason = f"Illegal route through region: '{req.entry}' -> '{req.exit}'."
            return
        
        @contextmanager
        def logger_ctx():
            rospy.loginfo(f"Reservation request from '{req.name}': {req.entry} -> {req.exit}")
            with self.reservation_lock: 
                yield
            if resp.success:
                rospy.loginfo(f"Reservation {resp.id} ({req.name}) approved.")
            else:
                rospy.loginfo(f"Reservation {resp.id} ({req.name}) recjected: {resp.reason}")
        
        with logger_ctx():
            try:
                earliest_entry = round(req.earliest_entry, 1)
                latest_entry = round(req.latest_entry, 1)
                
                offset = floor(earliest_entry) + (earliest_entry % self.TIME_STEP)
                time_ref += timedelta(seconds=offset)
                earliest_entry -= offset
                latest_entry -= offset
                assert 0 < earliest_entry, 'Negotiation Failed: Invalid window offsetting'

                max_window_entry = min(latest_entry - earliest_entry, 
                                    self.TIME_HORIZON - earliest_entry,
                                    self.MAX_WINDOW_ENTRY)
                max_window_entry = round(max_window_entry, 1)
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
                corridor = output['pass4']
                assert 0 < latest_entry - earliest_entry, 'Negotiation Faild: No time window to enter region'

            except AssertionError as e:
                msg, = e.args
                resp.reason = f'Reservation Error: {msg}'
                return
            else:
                meta = dict(id=resp.id,
                            name=req.name, 
                            time_ref=time_ref,
                            entry=req.entry, exit=req.exit,
                            earliest_entry=earliest_entry,
                            latest_entry=latest_entry,
                            earliest_exit=earliest_exit,
                            latest_exit=latest_exit)
                self.add_resevation(corridor=corridor, **meta)
                
                flat_corridor = shp.project_onto(corridor, 1, 2) <= 0

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

    def resolve_dangers(self, time_ref):
        dangers = []

        td_horizon = timedelta(seconds=self.TIME_HORIZON)
        
        for id, reservation in self.get_reservations():
            earliest_overlap = max(time_ref, reservation['time_ref'])
            latest_overlap = min(time_ref + td_horizon, reservation['time_ref'] + td_horizon)
            overlap = (latest_overlap - earliest_overlap).total_seconds()

            if not 0 < overlap:
                continue
            
            danger = np.ones(self.solver.timeline.shape + self.grid.shape)
            if time_ref < earliest_overlap:
                #  red:     [-----j----]
                # blue: [---i-----]
                i_offset = (earliest_overlap - time_ref).total_seconds()
                j_offset = (latest_overlap - reservation['time_ref']).total_seconds()
                i = ceil(i_offset / self.TIME_STEP)
                j = ceil(j_offset / self.TIME_STEP)
                danger[i:] = reservation['corridor'][:j]
            else: 
                #  red: [---i-----]
                # blue:     [-----j----]
                i_offset = (earliest_overlap - reservation['time_ref']).total_seconds()
                j_offset = (latest_overlap - time_ref).total_seconds()
                i = ceil(i_offset / self.TIME_STEP)
                j = ceil(j_offset / self.TIME_STEP)
                danger[:j] = reservation['corridor'][i:]
            dangers.append(danger)

        if not dangers:
            rospy.loginfo('Intersection free!')

        return dangers

    @service(SafeInput)
    def safe_input_srv(self, req, resp):
        resp.success = False
        resp.reason = 'Unknown.'
        
        try:
            time_ref = datetime.fromisoformat(req.time_ref)
        except Exception:
            resp.reason = f"Malformed ISO time: '{req.time_ref}'."
            return
        
        for id, reservation in self.get_reservations():
            if req.id == id: break
        else:
            resp.reason = f'Unknown reservation {req.id}'
            return
        
        idx = (np.array([req.x, req.y, req.a]) - self.grid.domain.lo[:3]) / np.array(self.grid.spacings)[:3]
        idx = np.where(self.grid._is_periodic_dim, idx % np.array(self.grid.shape), idx)
        idx = np.round(idx).astype(int)

        if (idx < 0).any() or (self.grid.shape[:3] <= idx).any():
            resp.reason = 'Outside grid'
            return
        
        ix, iy, ia = idx
        vf = reservation['corridor'][0:2, ix-1:ix+2, iy-1:iy+2, ia-1:ia+1, :, :]
        vf = shp.project_onto(vf, 0, 4, 5)
        vf = vf.max(axis=0)
        id, iv = np.unravel_index(vf.argmin(), vf.shape)

        resp.steering = self.grid.coordinate_vectors[4][id]
        resp.velocity = self.grid.coordinate_vectors[5][id]
        resp.success = True


    def run(self):
        rospy.spin()

if __name__ == '__main__':

    ## Start node ##
    Server().run()

