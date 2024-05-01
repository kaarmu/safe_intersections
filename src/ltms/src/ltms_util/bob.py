import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import pi, inf, ceil, floor
from pathlib import Path

import numpy as np
import hj_reachability as hj
import hj_reachability.shapes as shp

from .rc import RC
from .util import setdefaults
from .env import create_4way
from .solver import Solver

class Bob:

    @dataclass(frozen=True)
    class _Options:
        interactive: bool

        data_dir: str
        time_step: float
        time_horizon: float
        max_window: float
        
        model: str
        grid_shape: tuple[int, ...]
        min_bounds: list[float]
        max_bounds: list[float]

    def __init__(self, **kwargs):
        setdefaults(kwargs, RC['bob'])
        self.o = Bob._Options(**kwargs)

        self._reservations = {}

        grid_shape = self.o.grid_shape
        min_bounds = np.array(self.o.min_bounds)
        max_bounds = np.array(self.o.max_bounds)
        
        grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(min_bounds, max_bounds),
                                                                       grid_shape, periodic_dims=2)
 
        self.solver = Solver(grid=grid,
                             time_step=self.o.time_step,
                             time_horizon=self.o.time_horizon,
                             accuracy='medium',
                             dynamics=dict(cls=vars(hj.systems)[self.o.model],
                                           min_steer=-pi * 5/4,
                                           max_steer=+pi * 5/4,
                                           min_accel=-0.5,
                                           max_accel=+0.5),
                             interactive=self.o.interactive)
        
        self.load_env_and_routes()
            
    def load_env_and_routes(self):
        
        self.env = {}
        for loc in RC['locations']:
            filename = Path(self.o.data_dir) / f'G{self.solver.code_grid}-{loc}.npy'
            if filename.exists():
                self.env[loc] = np.load(filename, allow_pickle=True)
                if self.o.interactive: print(f'Loading {filename}')
            else:
                self.env.update(create_4way(self.solver.grid, loc))
                if self.o.interactive: print(f'Saving {filename}')
                np.save(filename, self.env[loc], allow_pickle=True)
        if self.o.interactive: print('Environment done.', end='\n\n')

        self.routes = {}
        for (entry, exit), locs in RC['permitted_routes'].items():
            code = (f'G{self.solver.code_grid}'
                    f'D{self.solver.code_dynamics}'
                    f'T{self.solver.code_time}')
            filename = Path(self.o.data_dir) / f'{code}-pass1-{entry}-{exit}.npy'
            if filename.exists():
                if self.o.interactive: print(f'Loading {filename}')
                self.routes[entry, exit] = np.load(filename, allow_pickle=True)
            else:
                constraints = shp.union(*[self.env[loc] for loc in locs])

                output = self.solver.run_analysis('pass1',
                                                  exit=self.env[exit],
                                                  constraints=constraints)
                
                if self.o.interactive: print(f'Saving {filename}')
                np.save(filename, output['pass1'], allow_pickle=True)
                self.routes[entry, exit] = output['pass1']
        if self.o.interactive: print('Offline analyses done.', end='\n\n')

    def resolve_dangers(self, time_ref):
        td_horizon = timedelta(seconds=self.o.time_horizon)

        dangers = []
        
        for _, reservation in self._reservations.items():
            earliest_overlap = max(time_ref, reservation['time_ref'])
            latest_overlap = min(time_ref + td_horizon, reservation['time_ref'] + td_horizon)
            overlap = (latest_overlap - earliest_overlap).total_seconds()

            if not 0 < overlap:
                continue
            
            danger = np.ones((26, 31, 31, 25, 7))
            if time_ref < earliest_overlap:
                # HPV:     [-----j----)
                # LPV: [---i-----)
                i_offset = (earliest_overlap - time_ref).total_seconds()
                j_offset = (latest_overlap - reservation['time_ref']).total_seconds()
                i = ceil(i_offset / self.o.time_step)
                j = ceil(j_offset / self.o.time_step)
                danger[i:i+j] = reservation['analysis']['pass4'][:j]
            else:# if time_ref > earliest_overlap: 
                # HPV: [---i-----)
                # LPV:     [-----j----)
                i_offset = (earliest_overlap - reservation['time_ref']).total_seconds()
                j_offset = (latest_overlap - time_ref).total_seconds()
                i = ceil(i_offset / self.o.time_step)
                j = ceil(j_offset / self.o.time_step)
                danger[:j] = reservation['analysis']['pass4'][i:i+j]
            dangers.append(danger)

        if self.o.interactive and not dangers:
            print('Intersection free!', end='\n\n')

        return dangers

    def reserve(self, time_ref, entry_loc, exit_loc, earliest_entry=0, latest_entry=inf, **solver_kwargs):
        setdefaults(solver_kwargs,
                    max_window=self.o.max_window)
        
        rid = secrets.token_hex(4)

        try:
            earliest_entry = round(max(earliest_entry, 0), 1)
            latest_entry = round(min(latest_entry, self.o.time_horizon), 1)
            
            offset = round(floor(earliest_entry) + (earliest_entry % self.o.time_step), 1)
            time_ref += timedelta(seconds=offset)
            earliest_entry -= offset
            latest_entry -= offset
            assert 0 <= earliest_entry <= latest_entry <= self.o.time_horizon, f'Negotiation Failed: Invalid window offsetting (offset={offset})'

            max_window_entry = round(min(latest_entry - earliest_entry, self.o.max_window), 1)
            assert max_window_entry, 'Negotiation Failed: Invalid entry window requested'
            
            dangers = self.resolve_dangers(time_ref)
            output = self.solver.run_analysis('pass4',
                                              pass1=self.routes[entry_loc, exit_loc],
                                              entry=self.env[entry_loc],
                                              exit=self.env[exit_loc],
                                              dangers=dangers,
                                              **solver_kwargs)
            
            earliest_entry = max(earliest_entry, output['earliest_entry'])
            latest_entry = min(latest_entry, output['latest_entry'])
            earliest_exit = output['earliest_exit']
            latest_exit = output['latest_exit']
            assert 0 < latest_entry - earliest_entry, 'Negotiation Faild: No time window to enter region'

        except AssertionError as e:
            msg, = e.args
            return {'success': False, 'reason': f'Reservation Error: {msg}'}
        else:
            self._reservations[rid] = dict(time_ref=time_ref,
                                           entry_loc=entry_loc, 
                                           exit_loc=exit_loc,
                                           earliest_entry=earliest_entry,
                                           latest_entry=latest_entry,
                                           earliest_exit=earliest_exit,
                                           latest_exit=latest_exit,
                                           analysis=output)
            
            out = {}
            out['id'] = rid
            out['time_ref'] = time_ref
            out['analysis'] = output
            out['success'] = True
            out['reason'] = ''

            return out
        
    def clean_all(self):
        for rid, _ in list(self._reservations.items()):
            self._reservations.pop(rid, None)
        
    def clean_from_time(self, time_ref):
        for rid, reservation in list(self._reservations.items()):
            if reservation['time_ref'] + timedelta(seconds=self.o.time_horizon) < time_ref:
                self._reservations.pop(rid, None)

    def reserve_many(self, *reqs):
        out = []
        for kwargs in reqs:
            r = self.reserve(**kwargs)
            assert r['success'], r['reason']
            out.append(r)
        return out
