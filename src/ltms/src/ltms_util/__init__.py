from time import time
from math import ceil
from struct import pack

import numpy as np
import numpy.lib.stride_tricks as st
import hj_reachability as hj
import hj_reachability.shapes as shp

def setdefaults(kw, *args, **kwargs):
    if len(args) == 0:
        assert kwargs, 'Missing arguments'
        defaults = kwargs
    elif len(args) == 1:
        defaults, = args
        assert isinstance(defaults, dict), 'Single-argument form must be default dictionary'
        assert not kwargs, 'Cannot supply keywords arguments with setdefault({...}) form'
    elif len(args) == 2:
        key, val = args
        assert not kwargs, 'Cannot supply keywords arguments with setdefault(key, val) form'
        defaults = {key: val}
    for key, val in defaults.items():
        kw.setdefault(key, val)

def new_timeline(target_time, start_time=0, time_step=0.2):
    assert time_step > 0
    is_forward = target_time >= start_time
    target_time += 1e-5 if is_forward else -1e-5
    time_step *= 1 if is_forward else -1
    return np.arange(start_time, target_time, time_step)

def find_windows(mask, N=1, M=None):
    """Find the indices of windows where at least N but less than M consequtive elements are true."""
    mask = np.asarray(mask)
    assert N <= len(mask)
    window_view = st.sliding_window_view(mask, window_shape=N)
    ix, = np.where(N == np.sum(window_view, axis=1))
    if M is not None:
        assert M <= len(mask)
        assert N < M
        window_view = st.sliding_window_view(mask, window_shape=M)
        jx = ix[ix + M <= len(mask)]
        iix, = np.where(np.sum(window_view[jx], axis=1) < M)
        ix = ix[iix]
    return ix

def earliest_window(mask, N=1, M=None):
    """Find the first window where at least N but less than M consequtive elements are true."""
    mask = np.asarray(mask)
    windows = find_windows(mask, N, M)
    if len(windows) == 0:
        return np.array([], int)
    i = windows[0] # Earliest window
    mask = mask[i:] if M is None else mask[i:i+M]
    for j, n in enumerate(mask.cumsum()[N-1:]):
        if n != j + N:
            break
    return np.arange(i, i+n)

def new_timeline(target_time, start_time=0, time_step=0.2):
    assert time_step > 0
    is_forward = target_time >= start_time
    target_time += 1e-5 if is_forward else -1e-5
    time_step *= 1 if is_forward else -1
    return np.arange(start_time, target_time, time_step)

class Solver:

    AVOID_MARGIN = 0

    def __init__(self, grid: hj.Grid, dynamics: dict, time_horizon: float, time_step: float,  accuracy='medium', interactive=True):

        self.grid = grid

        self.time_step = time_step
        self.time_horizon = time_horizon
        self.timeline = new_timeline(time_horizon, time_step=time_step)

        cls = dynamics.pop('cls')
        self.reach_dynamics = cls(**dynamics).with_mode('reach')
        self.avoid_dynamics = cls(**dynamics).with_mode('avoid')

        self.solver_settings = hj.SolverSettings.with_accuracy(accuracy)

        self.is_interactive = interactive

        # simple code checks to guard against different solver settings
        bs = pack('ff', self.time_horizon, self.time_step)
        self.code_time = bytes([sum(bs) % 256]).hex()

        bs = pack('i'*self.grid.ndim, *self.grid.shape)
        self.code_grid = bytes([sum(bs) % 256]).hex()

    def brs(self, times, target, constraints=None, *, mode='reach'):
        dynamics = self.reach_dynamics if mode == 'reach' else self.avoid_dynamics
        times = -np.asarray(times)
        if not  shp.is_invariant(self.grid, times, target):
            target = np.flip(target, axis=0)
        if not shp.is_invariant(self.grid, times, constraints):
            constraints = np.flip(constraints, axis=0)
        values = hj.solve(self.solver_settings, dynamics, self.grid,
                        times, target, constraints)
        values = np.asarray(values)
        values = np.flip(values, axis=0)
        return values

    def frs(self, times, target, constraints=None, *, mode='avoid'):
        times = np.asarray(times)
        dynamics = self.reach_dynamics if mode == 'reach' else self.avoid_dynamics
        values = hj.solve(self.solver_settings, dynamics, self.grid,
                        times, target, constraints)
        values = np.asarray(values)
        return values

    def run_analysis(self, *passes, **kwargs):
        """
        Required kwargs for different passes:        

            Pass 1:
                - exit: exit location
                - constraints: state constraints

            Pass 2:
                - pass1: output from Pass 1
                - dangers: list of danger time-state sets
                - exit: exit location

            Pass 3:
                - pass2: output from Pass 2
                - entry: entry location

            Pass 4:
                - pass3: output from Pass 3
                - exit: exit location

        Returns:
            - pass1, pass2, pass3, pass4: if requested
            - earliest_entry, latest_exit: if pass3 was computed
            - earliest_exit, latest_exit: if pass4 was computed
        """

        interactive = kwargs.pop('interactive', self.is_interactive)
        min_window = kwargs.pop('min_window', 1)  # Need at least 1 second to enter/exit
        max_window = kwargs.pop('max_window', 2)  # Allow up to 2 seconds to enter/exit
        min_window_entry = kwargs.pop('min_window_entry', min_window)
        max_window_entry = kwargs.pop('max_window_entry', max_window)
        min_window_exit = kwargs.pop('min_window_exit', min_window)
        max_window_exit = kwargs.pop('max_window_exit', max_window)

        ALL_PASSES = ['pass1', 'pass2', 'pass3', 'pass4']
        passes = passes or ALL_PASSES
        if 'all' in passes:
            passes = ALL_PASSES
        passes = [name for name in ALL_PASSES if name in passes]

        passes_out = list(passes)
        out = {}

        for name in passes:
            assert name in ALL_PASSES, f'Invalid pass: {name}'
            i = ALL_PASSES.index(name)
            
            if i > 0 and ALL_PASSES[i-1] not in passes + list(kwargs):
                passes += [ALL_PASSES[i-1]]
        passes_sch = [name for name in ALL_PASSES if name in passes]

        if interactive:
            msg = 'Running analysis with the following passes: '
            msg += ', '.join(passes_sch) + '\n'
            msg += '-' * (len(msg)-1)
            print(msg)

        def to_shared(values, **kwargs):
            return shp.project_onto(values, 0, 1, 2, **kwargs)
        def when_overlapping(a, b):
            return np.where(shp.project_onto(shp.intersection(a, b), 0) <= 0)[0]
        
        if 'pass1' in passes_sch:
            if interactive:
                print('\n', 'Pass 1: Initial BRS')
            
            rules = kwargs['constraints']
            end = kwargs['exit']

            target = shp.make_tube(self.timeline, end)
            constraints = rules

            start_time = time()
            output = self.brs(self.timeline, target, constraints)
            stop_time = time()

            if interactive:
                print(f'Time To Compute: {stop_time - start_time:.02f}')
            
            kwargs['pass1'] = output
            if 'pass1' in passes_out: out['pass1'] = output.copy()

        if 'pass2' in passes_sch:
            if interactive:
                print('\n', 'Pass 2: Avoidance')
            
            output = kwargs['pass1']
            avoid = kwargs['dangers']
            end = kwargs['exit']

            if avoid:
                avoid_target = shp.union(
                    *map(lambda target: to_shared(target, keepdims=True), avoid),
                ) - self.AVOID_MARGIN # increase avoid set with heuristic margin

                interact_window = when_overlapping(output, avoid_target)
                if 0 < interact_window.size:
                    i, j = interact_window[0], interact_window[-1] + 1

                    # Recompute solution until after last interaction
                    constraints = shp.setminus(output, avoid_target)[:j+1]
                    target = output[:j+1]    # last step is target to reach pass1
                    target[:-1] = end       # all other steps are recomputed to end
                    
                    start_time = time()
                    output[:j+1] = self.brs(self.timeline[:j+1], target, constraints)
                    stop_time = time()
                
                    if interactive:
                        print(f'Time To Compute: {stop_time - start_time:.02f}')
                        print(f'First Interaction: {self.timeline[i]:.01f}')
                        print(f'Last Interaction: {self.timeline[j-1]:.01f}')

            kwargs['pass2'] = output
            if 'pass2' in passes_out: out['pass2'] = output.copy()

        if 'pass3' in passes_sch:
            if interactive:
                print('\n', 'Pass 3: Planning, Departure')
            
            output = kwargs['pass2']
            start = kwargs['entry']

            min_nsteps = ceil(min_window_entry / self.time_step)
            max_nsteps = ceil(max_window_entry / self.time_step)
            depart_target = shp.intersection(output, start)
            depart_window = earliest_window(shp.project_onto(depart_target, 0) <= 0, min_nsteps)
            assert depart_window.size > 0, 'Analysis Failed: No time window to enter region'
            w0 = 0
            wn = min(max_nsteps+1, len(depart_window)-1)
            i = depart_window[w0] # Earliest entry index
            j = depart_window[wn] # Latest entry index
            depart_target[j:] = 1

            start_time = time()
            output[i:] = self.frs(self.timeline[i:], depart_target[i:], output[i:])
            output[:i] = 1 # Remove all values before departure
            stop_time = time()

            if interactive:
                print(f'Time To Compute: {stop_time - start_time:.02f}')
                print(f'Earliest Departure: {self.timeline[i]:.01f}')
                print(f'Latest Departure: {self.timeline[j-1]:.01f}')

            kwargs['pass3'] = output
            if 'pass3' in passes_out: out['pass3'] = output.copy()
            out['earliest_entry'] = self.timeline[i]
            out['latest_entry'] = self.timeline[j-1]

        if 'pass4' in passes_sch:
            if interactive:
                print('\n', 'Pass 4: Planning, Arrival')
            
            output = kwargs['pass3']
            end = kwargs['exit']

            min_nsteps = ceil(min_window_exit / self.time_step)
            max_nsteps = ceil(max_window_exit / self.time_step)
            arrival_target = shp.intersection(output, end)
            arrival_window = earliest_window(shp.project_onto(arrival_target, 0) <= 0, min_nsteps)
            assert arrival_window.size > 0, 'Analysis Failed: No time window to exit region'
            w0 = 0
            wn = min(max_nsteps+1, len(depart_window)-1)
            m = arrival_window[w0] # Earliest exit index
            n = arrival_window[wn] # Latest exit index
            arrival_target[n:] = 1

            start_time = time()
            output[i:n] = self.brs(self.timeline[i:n], arrival_target[i:n], output[i:n])
            output[n:] = 1 # Remove all values after arrival
            stop_time = time()

            if interactive:
                print(f'Time To Compute: {stop_time - start_time:.02f}')
                print(f'Earliest Arrival: {self.timeline[m]:.01f}')
                print(f'Latest Arrival: {self.timeline[n-1]:.01f}')

            kwargs['pass4'] = output
            if 'pass4' in passes_out: out['pass4'] = output.copy()
            out['earliest_exit'] = self.timeline[m]
            out['latest_exit'] = self.timeline[n-1]
        
        return out

    def run_many(self, *objectives, **kwargs):
        results = []
        for o in objectives:
            results += self.run_analysis('pass4', **o, **kwargs, avoid=results)
            print('\n')
        return results
    
def create_4way(grid, *envs):
    out = {}

    X, Y, A, D, V = range(grid.ndim)

    X0 = grid.domain.lo[X]
    Y0 = grid.domain.lo[Y]

    XN = grid.domain.hi[X] - grid.domain.lo[X]
    YN = grid.domain.hi[Y] - grid.domain.lo[Y]

    speedlimit = shp.rectangle(grid, axes=V, target_min=0.3, target_max=0.6)

    envs_out = tuple(envs)
    envs_sch = tuple(envs)

    # BIG OBS TO SELF: if entering from south then we're traveling in north direction
    # => => => 'entry_s' requires 'road_n'

    if {'entry_n', 'exit_s'} & set(envs_sch) and 'road_s' not in envs_sch:
        envs_sch += ('road_s',)
    if {'entry_e', 'exit_w'} & set(envs_sch) and 'road_w' not in envs_sch:
        envs_sch += ('road_w',)
    if {'entry_s', 'exit_n'} & set(envs_sch) and 'road_n' not in envs_sch:
        envs_sch += ('road_n',)
    if {'entry_w', 'exit_e'} & set(envs_sch) and 'road_e' not in envs_sch:
        envs_sch += ('road_e',)
    if {'road_s', 'road_w', 'road_n', 'road_e'} & set(envs_sch) and 'center' not in envs_sch:
        envs_sch += ('center',)

    ## CENTER ##

    if 'center' in envs_sch:
        out['center'] = shp.intersection(shp.hyperplane(grid, normal=[-1, -1], offset=[X0 + 0.25*XN, Y0 + 0.25*YN]),
                                         shp.hyperplane(grid, normal=[+1, -1], offset=[X0 + 0.75*XN, Y0 + 0.25*YN]),
                                         shp.hyperplane(grid, normal=[+1, +1], offset=[X0 + 0.75*XN, Y0 + 0.75*YN]),
                                         shp.hyperplane(grid, normal=[-1, +1], offset=[X0 + 0.25*XN, Y0 + 0.75*YN]),
                                         shp.rectangle(grid,
                                                       target_min=[X0 + 0.2*XN, Y0 + 0.2*YN],
                                                       target_max=[X0 + 0.8*XN, Y0 + 0.8*YN]))
        out['center'] = shp.intersection(out['center'], speedlimit)
    
    ## ROADS ##

    if 'road_e' in envs_sch:
        out['road_e'] = shp.rectangle(grid,
                                    axes=[Y, A],
                                    target_min=[Y0 + 0.3*YN, -np.pi/5],
                                    target_max=[Y0 + 0.5*YN, +np.pi/5])
        out['road_e'] = shp.union(shp.intersection(out['road_e'], speedlimit), out['center'])
    if 'road_w' in envs_sch:
        out['road_w'] = shp.rectangle(grid,
                                    axes=[Y, A],
                                    target_min=[Y0 + 0.5*YN, +np.pi - np.pi/5],
                                    target_max=[Y0 + 0.7*YN, -np.pi + np.pi/5])
        out['road_w'] = shp.union(shp.intersection(out['road_w'], speedlimit), out['center'])
    if 'road_n' in envs_sch:
        out['road_n'] = shp.rectangle(grid,
                                    axes=[X, A],
                                    target_min=[X0 + 0.5*XN, +np.pi/2 - np.pi/5],
                                    target_max=[X0 + 0.7*XN, +np.pi/2 + np.pi/5])
        out['road_n'] = shp.union(shp.intersection(out['road_n'], speedlimit), out['center'])
    if 'road_s' in envs_sch:
        out['road_s'] = shp.rectangle(grid,
                                    axes=[X, A],
                                    target_min=[X0 + 0.3*XN, -np.pi/2 - np.pi/5],
                                    target_max=[X0 + 0.5*XN, -np.pi/2 + np.pi/5])
        out['road_s'] = shp.union(shp.intersection(out['road_s'], speedlimit), out['center'])

    ## ENTRIES ##

    if 'entry_e' in envs_sch:
        out['entry_e']  = shp.rectangle(grid, 
                                        target_min=[X0 + 0.85*XN, Y0 + 0.53*YN], 
                                        target_max=[X0 + 1.00*XN, Y0 + 0.67*YN])
        out['entry_e']  = shp.intersection(out['entry_e'], out['road_w'])
    if 'entry_w' in envs_sch:
        out['entry_w']  = shp.rectangle(grid, 
                                        target_min=[X0 + 0.00*XN, Y0 + 0.33*YN], 
                                        target_max=[X0 + 0.15*XN, Y0 + 0.47*YN])
        out['entry_w']  = shp.intersection(out['entry_w'], out['road_e'])
    if 'entry_n' in envs_sch:
        out['entry_n']  = shp.rectangle(grid, 
                                        target_min=[X0 + 0.33*XN, Y0 + 0.85*YN], 
                                        target_max=[X0 + 0.47*XN, Y0 + 1.00*YN])
        out['entry_n']  = shp.intersection(out['entry_n'], out['road_s'])
    if 'entry_s' in envs_sch:
        out['entry_s']  = shp.rectangle(grid, 
                                        target_min=[X0 + 0.53*XN, Y0 + 0.00*YN], 
                                        target_max=[X0 + 0.67*XN, Y0 + 0.15*YN])
        out['entry_s']  = shp.intersection(out['entry_s'], out['road_n'])

    ## EXITS ##

    if 'exit_e' in envs_sch:
        out['exit_e']   = shp.rectangle(grid, 
                                        target_min=[X0 + 0.85*XN, Y0 + 0.33*YN], 
                                        target_max=[X0 + 1.00*XN, Y0 + 0.47*YN])
        out['exit_e']   = shp.intersection(out['exit_e'], out['road_e'])
    if 'exit_w' in envs_sch:
        out['exit_w']   = shp.rectangle(grid, 
                                        target_min=[X0 + 0.00*XN, Y0 + 0.53*YN], 
                                        target_max=[X0 + 0.15*XN, Y0 + 0.67*YN])
        out['exit_w']   = shp.intersection(out['exit_w'], out['road_w'])
    if 'exit_n' in envs_sch:
        out['exit_n']   = shp.rectangle(grid, 
                                        target_min=[X0 + 0.53*XN, Y0 + 0.85*YN], 
                                        target_max=[X0 + 0.67*XN, Y0 + 1.00*YN])
        out['exit_n']   = shp.intersection(out['exit_n'], out['road_n'])
    if 'exit_s' in envs_sch:
        out['exit_s']   = shp.rectangle(grid, 
                                        target_min=[X0 + 0.33*XN, Y0 + 0.00*YN], 
                                        target_max=[X0 + 0.47*XN, Y0 + 0.15*YN])
        out['exit_s']   = shp.intersection(out['exit_s'], out['road_s'])


    return {name: out[name] for name in envs_out}
