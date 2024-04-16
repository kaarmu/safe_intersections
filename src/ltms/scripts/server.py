#! /usr/bin/env python3.9

# Plain python imports
import numpy as np
import secrets
from math import ceil
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps

import rospy
from ltms.srv import ReserveCorridor
from geometry_msgs.msg import PoseStamped

import jax.numpy as jnp
from nav_msgs.msg import OccupancyGrid

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

    ENTRY_REGIONS = ['entry_s', 'entry_w', 'entry_n', 'entry_e']
    EXIT_REGIONS = ['exit_s', 'exit_w', 'exit_n', 'exit_e']
    REGIONS = [
        'center',
        'road_s', 'road_w', 'road_n', 'road_e',
        *ENTRY_REGIONS,
        *EXIT_REGIONS,
    ]

    PERMITTED_PASSTHROUGHS = {
        ('entry_s', 'exit_w'): ('road_s', 'road_w'),
        ('entry_s', 'exit_n'): ('road_s', 'road_n'),
        ('entry_s', 'exit_e'): ('road_s', 'road_e'),

        ('entry_w', 'exit_n'): ('road_w', 'road_n'),
        ('entry_w', 'exit_e'): ('road_w', 'road_e'),
        ('entry_w', 'exit_s'): ('road_w', 'road_s'),

        ('entry_n', 'exit_w'): ('road_n', 'road_w'),
        ('entry_n', 'exit_s'): ('road_n', 'road_s'),
        ('entry_n', 'exit_e'): ('road_n', 'road_e'),

        ('entry_e', 'exit_s'): ('road_e', 'road_s'),
        ('entry_e', 'exit_w'): ('road_e', 'road_w'),
        ('entry_e', 'exit_n'): ('road_e', 'road_n'),
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

        self.min_bounds = np.array([-1.2, -1.2, -np.pi, -np.pi/5, +0])
        self.max_bounds = np.array([+1.2, +1.2, +np.pi, +np.pi/5, +1])
        self.grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(self.min_bounds, self.max_bounds),
                                                                            (31, 31, 31, 7, 11),
                                                                            periodic_dims=2)

        self.solver = Solver(grid=self.grid, 
                             time_step=0.1,
                             time_horizon=self.TIME_HORIZON,
                             dynamics=dict(cls=hj.systems.SVEA5D,
                                           min_steer=-jnp.pi, max_steer=+jnp.pi,
                                           min_accel=-0.5, max_accel=+0.5),
                             interactive=False)

        self.environment = self.load_environment()
        self.offline_passes = self.load_offline_analyses()
        
        self.reservations = {}

        def clean_reservations_tmr(event):
            now = datetime.now().replace(microsecond=0)
            for id in self.reservations:
                if self.reservations[id]['origin_time'] + timedelta(seconds=self.TIME_HORIZON) < now:
                    del self.reservations[id]
        rospy.Timer(rospy.Duration(1), clean_reservations_tmr)

        ## Advertise services

        rospy.Service('~reserve_corridor', ReserveCorridor, self.reserve_corridor_srv)

        ## Node initialized

        rospy.loginfo(f'{self.__class__.__name__} initialized!')

    def load_environment(self):
        out = {}
        for env in self.REGIONS:
            filename = self.DATA_DIR / f'{env}.npy'
            if filename.exists():
                out[env] = np.load(filename, allow_pickle=True)
                print(f'Loading {filename}')
            else:
                out[env] = create_4way(self.grid, env)
                print(f'Saving {filename}')
                np.save(filename, out[env], allow_pickle=True)
        print('Environment done.')
        return out

    def load_offline_analyses(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        out = {}
        for (entry, exit), roads in self.PERMITTED_PASSTHROUGHS.items():
            filename = self.DATA_DIR / f'pass1-{entry}-{exit}.npy'
            if filename.exists():
                print(f'Loading {filename}')
                out[entry, exit] = np.load(filename, allow_pickle=True)
            else:
                constraints = shp.union(*[self.environment[road] for road in roads])

                pass1 = self.solver.run_analysis('pass1',
                                                 exit=self.environment[exit],
                                                 constraints=constraints)
                
                print(f'Saving {filename}')
                np.save(filename, pass1, allow_pickle=True)
                out[entry, exit] = pass1
        
        print('Offline analyses done.')
        return out
    
    @service(ReserveCorridor)
    def reserve_corridor_srv(self, req, resp):
        resp.success = False
        resp.reason = 'Unknown.'
        resp.id = secrets.token_hex(4)

        try:
            origin_time = datetime.fromisoformat(req.isotime).replace(microsecond=0)
        except Exception:
            resp.reason = f'Malformed ISO time: {req.isotime}'
            return
        
        if req.entry not in self.ENTRY_REGIONS:
            resp.reason = f'Illegal entry region: "{req.entry}".'
            return
        if req.exit not in self.EXIT_REGIONS:
            resp.reason = f'Illegal exit region: "{req.exit}".'
            return
        if (req.entry, req.exit) not in self.PERMITTED_PASSTHROUGHS:
            resp.reason = f'Illegal route through region: "{req.entry}" -> "{req.exit}".'
            return
        
        req.earliest_entry = round(req.earliest_entry, 1)
        req.latest_entry = round(req.latest_entry, 1)

        pass1 = self.offline_passes[req.entry, req.exit]

        try:
            dangers = self.find_dangers(origin_time + timedelta(seconds=req.earliest_entry))
            output = self.solver.run_analysis(pass1=pass1,
                                              timeline=self.timeline,
                                              entry=req.entry, exit=req.exit,
                                              danger=dangers)
        except AssertionError as e:
            msg, = e.args
            resp.reason = f'Reservation failed: {msg}'
            return
        else:
            self.reservations[resp.id] = dict(origin_time=origin_time,
                                              name=req.name, 
                                              entry=req.entry, exit=req.exit,
                                              earliest_entry=output['entry_window'][0],
                                              latest_entry=output['entry_window'][1],
                                              earliest_exit=output['exit_window'][0],
                                              latest_exit=output['exit_window'][1],
                                              corridor=output['pass4'])
            
            resp.isotime = origin_time.isoformat()
            resp.earliest_entry=output['entry_window'][0]
            resp.latest_entry=output['entry_window'][1]
            resp.earliest_exit=output['exit_window'][0]
            resp.latest_exit=output['exit_window'][1]
            resp.success = True

    def find_dangers(self, origin_time):
        td_horizon = timedelta(seconds=self.TIME_HORIZON)
        dangers = []
        for reservation in self.reservations.values():
            earliest_overlap = max(origin_time, reservation['origin_time'])
            latest_overlap = min(origin_time + td_horizon, reservation['origin_time'] + td_horizon)
            overlap = (latest_overlap - earliest_overlap).total_seconds()

            if not 0 < overlap:
                continue
            
            danger = np.ones(self.grid.shape)
            if origin_time < earliest_overlap:
                #  red:     [-----j----]
                # blue: [---i-----]
                i_offset = (earliest_overlap - origin_time).total_seconds()
                j_offset = (latest_overlap - reservation['origin_time']).total_seconds()
                i = ceil(i_offset / self.solver.time_step)
                j = ceil(j_offset / self.solver.time_step)
                danger[i:] = reservation['corridor'][:j]
            else: 
                #  red: [---i-----]
                # blue:     [-----j----]
                i_offset = (earliest_overlap - reservation['origin_time']).total_seconds()
                j_offset = (latest_overlap - origin_time).total_seconds()
                i = ceil(i_offset / self.solver.time_step)
                j = ceil(j_offset / self.solver.time_step)
                danger[:j] = reservation['corridor'][i:]
            dangers.append(danger)
        return dangers

    def run(self):
        rospy.spin()

class State(object):
    def __init__(self):
        self.x = -2
        self.y = -2
        self.yaw = pi/4
        self.v = 0.8

if __name__ == '__main__':

    ## Start node ##
    Server().run()

