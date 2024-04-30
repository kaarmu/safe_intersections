from time import time
from math import ceil
from struct import pack
from functools import wraps

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

def flatten(seq):
    for itm in seq:
        if isinstance(itm, type(seq)):
            yield from itm
        else:
            yield itm

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

        bs = pack('s' + 'ffi'*self.grid.ndim,
                  accuracy.encode(), 
                  *flatten([[self.grid.domain.lo[i], self.grid.domain.hi[i], self.grid.shape[i]]
                            for i in range(self.grid.ndim)]))
        self.code_grid = bytes([sum(bs) % 256]).hex()

        bs = bytes()
        bs += pack('sx', cls.__name__.encode())
        for key, val in dynamics.items():
            bs += pack('sxf', key.encode(), val)
        self.code_dynamics = bytes([sum(bs) % 256]).hex()

    def brs(self, times, target, constraints=None, *, mode='reach', interactive=True):
        jnp = hj.solver.jnp
        times = -jnp.asarray(times)
        target = jnp.asarray(target)
        constraints = jnp.asarray(constraints)
        if not  shp.is_invariant(self.grid, times, target):
            target = jnp.flip(target, axis=0)
        if not shp.is_invariant(self.grid, times, constraints):
            constraints = jnp.flip(constraints, axis=0)
        values = hj.solve(self.solver_settings, 
                          (self.reach_dynamics if mode == 'reach' else 
                           self.avoid_dynamics), 
                          self.grid,
                          times, 
                          target, 
                          constraints,
                          progress_bar=interactive)
        values = jnp.flip(values, axis=0)
        return np.asarray(values)

    def frs(self, times, target, constraints=None, *, mode='avoid', interactive=True):
        jnp = hj.solver.jnp
        times = jnp.asarray(times)
        target = jnp.asarray(target)
        constraints = jnp.asarray(constraints)
        values = hj.solve(self.solver_settings,
                          (self.reach_dynamics if mode == 'reach' else 
                           self.avoid_dynamics),
                          self.grid,
                          times,
                          target,
                          constraints,
                          progress_bar=interactive)
        return np.asarray(values)

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

        debug = kwargs.pop('debug', False)
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

        passes_out = ALL_PASSES if debug else list(passes)
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
            output = self.brs(self.timeline, target, constraints, interactive=interactive)
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
                    output[:j+1] = self.brs(self.timeline[:j+1], target, constraints, interactive=interactive)
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
            output[i:] = self.frs(self.timeline[i:], depart_target[i:], output[i:], interactive=interactive)
            output[:i] = 1 # Remove all values before departure
            stop_time = time()

            if interactive:
                print(f'Time To Compute: {stop_time - start_time:.02f}')
                print(f'Earliest Departure: {self.timeline[i]:.01f}')
                print(f'Latest Departure: {self.timeline[j-1]:.01f}')

            kwargs['pass3'] = output
            
            out['earliest_entry'] = self.timeline[i]
            out['latest_entry'] = self.timeline[j-1]
            if 'pass3' in passes_out: out['pass3'] = output.copy()

            if debug: out['depart_target'] = depart_target

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
            wn = min(max_nsteps+1, len(arrival_window)-1)
            m = arrival_window[w0] # Earliest exit index
            n = arrival_window[wn] # Latest exit index
            arrival_target[n:] = 1

            start_time = time()
            output[i:n] = self.brs(self.timeline[i:n], arrival_target[i:n], output[i:n], interactive=interactive)
            output[n:] = 1 # Remove all values after arrival
            stop_time = time()

            if interactive:
                print(f'Time To Compute: {stop_time - start_time:.02f}')
                print(f'Earliest Arrival: {self.timeline[m]:.01f}')
                print(f'Latest Arrival: {self.timeline[n-1]:.01f}')

            kwargs['pass4'] = output

            out['earliest_exit'] = self.timeline[m]
            out['latest_exit'] = self.timeline[n-1]
            if 'pass4' in passes_out: out['pass4'] = output.copy()

            if debug: out['arrival_target'] = arrival_target

        return out

    def run_many(self, *objectives, **kwargs):
        results = []
        for o in objectives:
            results += self.run_analysis('pass4', **o, **kwargs, avoid=results)
            print('\n')
        return results
    
def create_4way(grid, *envs):
    out = {}

    X, Y, A, V = range(grid.ndim)

    X0 = grid.domain.lo[X]
    Y0 = grid.domain.lo[Y]

    XN = grid.domain.hi[X] - grid.domain.lo[X]
    YN = grid.domain.hi[Y] - grid.domain.lo[Y]

    speedlimit = shp.rectangle(grid, axes=V, target_min=0.3, target_max=0.6)

    if not envs:
        envs += ('entry_n', 'entry_e', 'entry_s', 'entry_w')
        envs += ('exit_n', 'exit_e', 'exit_s', 'exit_w')
        envs += ('road_n', 'road_e', 'road_s', 'road_w')
        envs += ('center',)

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

import sys
if sys.version_info.minor >= 10:

    from pathlib import Path
    import matplotlib.pyplot as plt

    REG = {}

    REG['min_bounds'] = np.array([-1.2, -1.2, -np.pi, -np.pi/5, +0])
    REG['max_bounds'] = np.array([+1.2, +1.2, +np.pi, +np.pi/5, +1])
    REG['grid_shape'] = (31, 31, 25, 7, 7)

    REG['min_bounds'] = np.array([-1.2, -1.2, -np.pi, +0])
    REG['max_bounds'] = np.array([+1.2, +1.2, +np.pi, +1])
    REG['grid_shape'] = (31, 31, 25, 7)

    REG['extent'] = [REG['min_bounds'][0], REG['max_bounds'][0], 
                     REG['min_bounds'][1], REG['max_bounds'][1]]
    REG['bgpath'] = str((Path(__file__) / '../../data/4way.png').resolve())

    def auto_ax(f):
        @wraps(f)
        def wrapper(*args, ax: plt.Axes = None, **kwargs):
            if ax is None:
                _, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(9*4/3, 9))
            kwargs.update(ax=ax)
            return f(*args, **kwargs)
        return wrapper

    @auto_ax
    def plot_im(im, *, ax, transpose=False, **kwargs):
        setdefaults(kwargs, cmap='Blues', aspect='auto')
        if transpose:
            im = np.transpose(im)
        return [ax.imshow(im, origin='lower', **kwargs)]
    
    @auto_ax
    def plot_levels(vf, *, ax, **kwargs):
        return [ax.contourf(vf, **kwargs)]

    @auto_ax
    def plot_levelset(vf, **kwargs):
        setdefaults(kwargs, aspect='equal', cval=0.5)
        cval = kwargs.pop('cval')
        vf = np.where(vf <= 0, cval, np.nan)
        kwargs.update(vmin=0, vmax=1,)
        return plot_im(vf, **kwargs)

    @auto_ax
    def plot_levelset_many(*vfs, **kwargs):
        out = []
        f = lambda itm: itm if isinstance(itm, tuple) else (itm, {})
        for vf, kw in map(f, vfs):
            setdefaults(kw, kwargs)
            out += plot_levelset(vf, **kw)
        return out
    
    @auto_ax
    def plot_tlrc(vf, **kwargs):
        print('Assumes Bicycle4D')
        dt = 0.2
        shape = 31, 31, 25, 7
        x, y, h, v = np.meshgrid(*[np.linspace(min_bounds[i], max_bounds[i], shape[i])
                                   for i in range(4)])
        f = np.array([
            v * np.cos(h),
            v * np.sin(h),
            np.zeros_like(h),
            np.zeros_like(v),
        ])

        g = np.array([
            [0],
            [0],
            [0],
            [1],
        ])

        out = []

        n = vf.shape[0]
        for i in range(n):
            idx = np.where(vf[i] <= 0)
            dvdx = np.array(np.gradient(vf[i]))[(...,) + idx]

            a = vf[i][idx] + dt*np.sum(dvdx * f[(...,) + idx], axis=0)
            b = (g.T @ dvdx)

            im = a + b * np.linspace(-0.4, 0.4) <= 0
            im.logic
            # out += plot_im(, **kwargs)

        return out 

    def new_map(*vfs, **kwargs):
        setdefaults(kwargs, 
                    bgpath=REG.get('bgpath', None),
                    alpha=0.9,
                    extent=REG.get('extent', None))
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(9*4/3, 9))
        ax.set_ylabel("y [m]")
        ax.set_xlabel("x [m]")
        ax.invert_yaxis()
        bgpath = kwargs.pop('bgpath')
        if bgpath is not None:
            ax.imshow(plt.imread(bgpath), extent=kwargs['extent'])
        plot_levelset_many(*vfs, **kwargs, ax=ax, transpose=True)
        fig.tight_layout()
        return fig
    
    load_plot = lambda name, *args: new_map(shp.project_onto(np.load(name), *args)).show()

    import plotly.graph_objects as go
    import skimage.io as sio

    def add_surface(*args, **kwargs):
        setdefaults(kwargs,
                    axes=(0, 1, 2),
                    min_bounds=REG.get('min_bounds', None),
                    max_bounds=REG.get('max_bounds', None),
                    grid_shape=REG.get('grid_shape', None),
                    colorscale='blues')

        axes = kwargs.pop('axes')
        max_bounds = kwargs.pop('max_bounds', None)
        min_bounds = kwargs.pop('min_bounds', None)
        grid_shape = kwargs.pop('grid_shape', None)

        setdefaults(kwargs,
                    {name: np.linspace(min_bounds[i], max_bounds[i], grid_shape[i])
                     for i, name in zip(axes, 'xs ys zs'.split())})

        xs, ys, zs = [kwargs.pop(name) for name in 'xs ys zs'.split()]

        data = []

        f = lambda itm: itm if isinstance(itm, (tuple, list)) else (itm, {})
        for vf, kw in map(f, args):
            setdefaults(kw, **kwargs)

            data += [
                go.Isosurface(
                    x=xs, y=ys, z=zs,
                    value=shp.project_onto(vf, *map(lambda v: v+1, axes)).flatten(),
                    showscale=False,
                    isomin=0,
                    surface_count=1,
                    isomax=0,
                    caps=dict(x_show=True, y_show=True),
                    **kw,
                ),
            ]
        
        return data

    def interact_tubes_time(times, *triplets, **kwargs):
        setdefaults(kwargs,
                    eye=None,
                    bgpath=REG.get('bgpath', None),
                    min_bounds=REG.get('min_bounds', None),
                    max_bounds=REG.get('max_bounds', None),
                    grid_shape=REG.get('grid_shape', None))

        eye = kwargs['eye']
        bgpath = kwargs['bgpath']
        min_bounds = kwargs['min_bounds']
        max_bounds = kwargs['max_bounds']
        grid_shape = kwargs['grid_shape']
        assert grid_shape is not None

        if bgpath is not None:
            background = sio.imread(kwargs['bgpath'], as_gray=True)
            background = np.flipud(background)
        
        bgaspect = (max_bounds[0]-min_bounds[0])/(max_bounds[1]-min_bounds[1])

        def render_frame(time_idx=None):
            data = []

            meshgrid = np.mgrid[times[0]:times[-1]:complex(0, len(times)),
                                min_bounds[0]:max_bounds[0]:complex(0, grid_shape[0]), 
                                min_bounds[1]:max_bounds[1]:complex(0, grid_shape[1])]
            
            if bgpath is not None:
                data += [
                    go.Surface(
                        x=np.linspace(min_bounds[0], max_bounds[0], background.shape[1]),
                        y=np.linspace(min_bounds[1], max_bounds[1], background.shape[0]),
                        z=times[0]*np.ones_like(background)-0.1,
                        surfacecolor=background,
                        colorscale='gray', 
                        showscale=False,
                    ),
                ]

            for triplet in triplets:

                colorscale, values = triplet[:2]
                kwargs = triplet[2] if len(triplet) == 3 else {}

                vf = shp.project_onto(values, 0, 1, 2)
                if time_idx is not None and time_idx < len(times)-1:
                    vf[time_idx+1:] = 1

                data += [
                    go.Isosurface(
                        x=meshgrid[1].flatten(),
                        y=meshgrid[2].flatten(),
                        z=meshgrid[0].flatten(),
                        value=vf.flatten(),
                        colorscale=colorscale,
                        showscale=False,
                        isomin=0,
                        surface_count=1,
                        isomax=0,
                        caps=dict(x_show=True, y_show=True),
                        **kwargs,
                    ),
                ]
            
            fw = go.Figure(data=data)
            fw.layout.update(# width=720, height=720, 
                            margin=dict(l=10, r=10, t=10, b=10),
                            #  legend=dict(yanchor='bottom', xanchor='left', x=0.05, y=0.05, font=dict(size=16)),
                            scene=dict(xaxis_title='x [m]',
                                        yaxis_title='y [m]',
                                        zaxis_title='t [s]',
                                        aspectratio=dict(x=bgaspect, y=1, z=3/4)),
                            scene_camera=dict(eye=eye))
            fw._config = dict(toImageButtonOptions=dict(height=720, width=720, scale=6))
            return fw
        return render_frame

    def interact_tubes_axis(times, *triplets, **kwargs):
        setdefaults(kwargs,
                    eye=None,
                    xaxis=0, yaxis=1, zaxis=2,
                    bgpath=REG.get('bgpath', None),
                    min_bounds=REG.get('min_bounds', None),
                    max_bounds=REG.get('max_bounds', None),
                    grid_shape=REG.get('grid_shape', None))

        eye = kwargs['eye']
        min_bounds = kwargs['min_bounds']
        max_bounds = kwargs['max_bounds']
        grid_shape = kwargs['grid_shape']
        assert grid_shape is not None

        bgaspect = (max_bounds[0]-min_bounds[0])/(max_bounds[1]-min_bounds[1])

        bgpath = kwargs['bgpath']
        if bgpath is not None:
            background = sio.imread(kwargs['bgpath'], as_gray=True)
            background = np.flipud(background)

        xaxis, yaxis, zaxis = kwargs['xaxis'], kwargs['yaxis'], kwargs['zaxis']
        proj_target = [i for i in (xaxis, yaxis, zaxis) if isinstance(i, int)]
        if isinstance(xaxis, int):
            xaxis = np.linspace(min_bounds[xaxis], max_bounds[xaxis], grid_shape[xaxis])
        if isinstance(yaxis, int):
            yaxis = np.linspace(min_bounds[yaxis], max_bounds[yaxis], grid_shape[yaxis])    
        if isinstance(zaxis, int):
            zaxis = np.linspace(min_bounds[zaxis], max_bounds[zaxis], grid_shape[zaxis])    

        def render_frame(time_idx=None):
            data = []

            meshgrid = np.meshgrid(xaxis, yaxis, zaxis)
            
            if bgpath is not None:
                data += [
                    go.Surface(
                        x=np.linspace(min_bounds[0], max_bounds[0], background.shape[1]),
                        y=np.linspace(min_bounds[1], max_bounds[1], background.shape[0]),
                        z=zaxis.min()*np.ones_like(background)-0.1,
                        surfacecolor=background,
                        colorscale='gray', 
                        showscale=False,
                    ),
                ]

            for triplet in triplets:

                colorscale, values = triplet[:2]
                kwargs = triplet[2] if len(triplet) == 3 else {}

                vf = (shp.project_onto(values, *proj_target) if time_idx is None else
                    shp.project_onto(values, 0, *proj_target)[time_idx])

                data += [
                    go.Isosurface(
                        x=meshgrid[0].flatten(),
                        y=meshgrid[1].flatten(),
                        z=meshgrid[2].flatten(),
                        value=vf.flatten(),
                        colorscale=colorscale,
                        showscale=False,
                        isomin=0,
                        surface_count=1,
                        isomax=0,
                        caps=dict(x_show=True, y_show=True),
                        **kwargs,
                    ),
                ]
            
            fw = go.Figure(data=data)
            fw.layout.update(width=720, height=720, 
                            margin=dict(l=10, r=10, t=10, b=10),
                            #  legend=dict(yanchor='bottom', xanchor='left', x=0.05, y=0.05, font=dict(size=16)),
                            scene=dict(xaxis_title='x [m]',
                                       yaxis_title='y [m]',
                                       zaxis_title=['Yaw [rad]', 'Delta [rad]', 'Vel [m/s]', 'FIXME'][-1],
                                       aspectratio=dict(x=bgaspect, y=1, z=3/4)),
                            scene_camera=dict(eye=eye))
            fw._config = dict(toImageButtonOptions=dict(height=720, width=720, scale=6))
            return fw
        return render_frame

    def interact_tubes(*args, axis=None, **kwargs):
        return (interact_tubes_time(*args, **kwargs) if axis is None else
                interact_tubes_axis(*args, **kwargs, axis=axis))

    def sphere_to_cartesian(r, theta, phi):
        theta *= np.pi/180
        phi *= np.pi/180
        return dict(x=r*np.sin(theta)*np.cos(phi),
                    y=r*np.sin(theta)*np.sin(phi),
                    z=r*np.cos(theta))
    
    def extended_plot():
        import json
        from datetime import datetime

        svea0 = np.load('../data/svea0.npy', allow_pickle=True)
        svea1 = np.load('../data/svea1.npy', allow_pickle=True)
        svea2 = np.load('../data/svea2.npy', allow_pickle=True)
        with open('../data/svea0.json') as f: svea0_time = datetime.fromisoformat(json.load(f)['time_ref'])
        with open('../data/svea1.json') as f: svea1_time = datetime.fromisoformat(json.load(f)['time_ref'])
        with open('../data/svea2.json') as f: svea2_time = datetime.fromisoformat(json.load(f)['time_ref'])
        diff = (svea2_time - svea0_time).total_seconds()
        n = int(np.ceil(diff / 0.2))
        timeline = new_timeline(0.2*n + 5)
        svea0_full = np.ones(timeline.shape + svea0.shape[1:])
        svea1_full = np.ones(timeline.shape + svea1.shape[1:])
        svea2_full = np.ones(timeline.shape + svea2.shape[1:])
        svea0_full[:26] = svea0
        n = int(np.ceil((svea1_time - svea0_time).total_seconds() / 0.2))
        svea1_full[n:n+26] = svea1
        n = int(np.ceil((svea2_time - svea0_time).total_seconds() / 0.2))
        svea2_full[n:n+26] = svea2
        interact_tubes(timeline, ('blues', svea0_full), ('reds', svea1_full), ('purples', svea2_full))().write_html('plot.html')

    def val2ind(x, axis, **kwargs):
        setdefaults(kwargs,
                    min_bounds=REG.get('min_bounds', None),
                    max_bounds=REG.get('max_bounds', None),
                    grid_shape=REG.get('grid_shape', None))
        min_bounds = kwargs.pop('min_bounds')
        max_bounds = kwargs.pop('max_bounds')
        grid_shape = kwargs.pop('grid_shape')

        dx = (max_bounds[axis] - min_bounds[axis]) / grid_shape[axis]
        return round((x - min_bounds[axis]) / dx) - 1
    
    def ind2val(i, axis, **kwargs):
        setdefaults(kwargs,
                    min_bounds=REG.get('min_bounds', None),
                    max_bounds=REG.get('max_bounds', None),
                    grid_shape=REG.get('grid_shape', None))
        min_bounds = kwargs.pop('min_bounds')
        max_bounds = kwargs.pop('max_bounds')
        grid_shape = kwargs.pop('grid_shape')

        dx = (max_bounds[axis] - min_bounds[axis]) / grid_shape[axis]
        return i*dx + min_bounds[axis]
    
    def create_simple_solver(**kwargs):
        from pathlib import Path
        import jax.numpy as jnp

        TIME_STEP = kwargs.pop('time_step', 0.2)
        TIME_HORIZON = kwargs.pop('time_horizon', 5)
        MAX_WINDOW_ENTRY = kwargs.pop('max_window_entry', 2)
        DATA_DIR = Path(kwargs.pop('data_dir', '../data')).resolve()

        ENTRY_LOCATIONS = ['entry_s', 'entry_w', 'entry_n', 'entry_e']
        EXIT_LOCATIONS = ['exit_s', 'exit_w', 'exit_n', 'exit_e']
        LOCATIONS = [
            'center',
            'road_s', 'road_w', 'road_n', 'road_e',
            *ENTRY_LOCATIONS,
            *EXIT_LOCATIONS,
        ]
        PERMITTED_ROUTES = {
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
        
        min_bounds = np.array([-1.2, -1.2, -np.pi, +0.3])
        max_bounds = np.array([+1.2, +1.2, +np.pi, +0.8])
        grid_shape = (31, 31, 25, 7)
        model = hj.systems.Bicycle4D

        grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(min_bounds, max_bounds),
                                                                       grid_shape,
                                                                       periodic_dims=2)
 
        solver = Solver(grid=grid,
                        time_step=TIME_STEP,
                        time_horizon=TIME_HORIZON,
                        accuracy='medium',
                        dynamics=dict(cls=model,
                                      min_steer=-jnp.pi * 5/4,
                                      max_steer=+jnp.pi * 5/4,
                                      min_accel=-0.5,
                                      max_accel=+0.5),
                        interactive=True)
        
        env = {}
        for loc in LOCATIONS:
            filename = DATA_DIR / f'G{solver.code_grid}-{loc}.npy'
            if filename.exists():
                env[loc] = np.load(filename, allow_pickle=True)
                print(f'Loading {filename}')
            else:
                env.update(create_4way(grid, loc))
                print(f'Saving {filename}')
                np.save(filename, env[loc], allow_pickle=True)
        print('Environment done.')

        routes = {}
        for (entry, exit), locs in PERMITTED_ROUTES.items():
            code = (f'G{solver.code_grid}'
                    f'D{solver.code_dynamics}'
                    f'T{solver.code_time}')
            filename = DATA_DIR / f'{code}-pass1-{entry}-{exit}.npy'
            if filename.exists():
                print(f'Loading {filename}')
                routes[entry, exit] = np.load(filename, allow_pickle=True)
            else:
                constraints = shp.union(*[env[loc] for loc in locs])

                output = solver.run_analysis('pass1',
                                             exit=env[exit],
                                             constraints=constraints)
                
                print(f'Saving {filename}')
                np.save(filename, output['pass1'], allow_pickle=True)
                routes[entry, exit] = output['pass1']
        print('Offline analyses done.')

        def run(entry, exit, **kwargs):
            setdefaults(kwargs,
                        max_window_entry=MAX_WINDOW_ENTRY,
                        pass1=routes[entry, exit],
                        dangers=[])
            return solver.run_analysis('pass4', entry=env[entry], exit=env[exit], **kwargs)
        
        return run
    
    def create_reservation_system(**kwargs):
        setdefaults(kwargs,
                    time_step=0.2,
                    time_horizon=5,
                    max_window_entry=2)
        
        from datetime import datetime, timedelta
        from math import floor

        simple_solver = create_simple_solver(**kwargs)

        reservations = {}

        TIME_STEP = kwargs.pop('time_step')
        TIME_HORIZON = kwargs.pop('time_horizon')
        MAX_WINDOW_ENTRY = kwargs.pop('max_window_entry')

        def resolve_dangers(time_ref):
            dangers = []

            td_horizon = timedelta(seconds=TIME_HORIZON)
            
            for id, reservation in reservations.items():
                earliest_overlap = max(time_ref, reservation['time_ref'])
                latest_overlap = min(time_ref + td_horizon, reservation['time_ref'] + td_horizon)
                overlap = (latest_overlap - earliest_overlap).total_seconds()

                if not 0 < overlap:
                    continue
                
                danger = np.ones((26, 31, 31, 25, 7))
                if time_ref < earliest_overlap:
                    #  red:     [-----j----]
                    # blue: [---i-----]
                    i_offset = (earliest_overlap - time_ref).total_seconds()
                    j_offset = (latest_overlap - reservation['time_ref']).total_seconds()
                    i = ceil(i_offset / TIME_STEP)
                    j = ceil(j_offset / TIME_STEP)
                    danger[i:] = reservation['corridor'][:j]
                else: 
                    #  red: [---i-----]
                    # blue:     [-----j----]
                    i_offset = (earliest_overlap - reservation['time_ref']).total_seconds()
                    j_offset = (latest_overlap - time_ref).total_seconds()
                    i = ceil(i_offset / TIME_STEP)
                    j = ceil(j_offset / TIME_STEP)
                    danger[:j] = reservation['corridor'][i:]
                dangers.append(danger - 0.1)

            if not dangers:
                print('Intersection free!')

            return dangers

        def reserve(time_ref, entry, exit, earliest_entry=0.5, latest_entry=2.5, **kwargs):

            try:
                earliest_entry = round(earliest_entry, 1)
                latest_entry = round(latest_entry, 1)
                
                offset = floor(earliest_entry) + (earliest_entry % TIME_STEP)
                time_ref += timedelta(seconds=offset)
                earliest_entry -= offset
                latest_entry -= offset
                assert 0 < earliest_entry, f'Negotiation Failed: Invalid window offsetting (offset={offset})'

                max_window_entry = min(latest_entry - earliest_entry, 
                                       TIME_HORIZON - earliest_entry,
                                       MAX_WINDOW_ENTRY)
                max_window_entry = round(max_window_entry, 1)
                assert max_window_entry, 'Negotiation Failed: Invalid entry window requested'
                
                dangers = resolve_dangers(time_ref)
                output = simple_solver(entry, exit, dangers=dangers, **kwargs)
                
                earliest_entry = max(earliest_entry, output['earliest_entry'])
                latest_entry = min(latest_entry, output['latest_entry'])
                earliest_exit = output['earliest_exit']
                latest_exit = output['latest_exit']
                corridor = output['pass4']
                assert 0 < latest_entry - earliest_entry, 'Negotiation Faild: No time window to enter region'

            except AssertionError as e:
                msg, = e.args
                return {'success': False, 'reason': f'Reservation Error: {msg}'}
            else:
                reservations[id] = dict(time_ref=time_ref,
                                        entry=entry, exit=exit,
                                        earliest_entry=earliest_entry,
                                        latest_entry=latest_entry,
                                        earliest_exit=earliest_exit,
                                        latest_exit=latest_exit,
                                        corridor=corridor)
                
                out = {}
                out['id'] = id
                out['time_ref'] = time_ref
                out['analysis'] = output
                out['success'] = True
                out['reason'] = ''

                return out
            
        def clean_from(time_ref):
            for id, reservation in reservations.items():
                if reservation['time_ref'] + timedelta(seconds=TIME_HORIZON) < time_ref:
                    reservations.pop(id, None)

        return reserve, clean_from
    
    def figs():
        from datetime import datetime, timedelta
        import matplotlib as mpl

        mpl.rcParams.update(**{'font.size': 22})

        now = datetime.now()

        reserve, clean_from = create_reservation_system()

        v1 = reserve(now, 'entry_s', 'exit_n', max_window=1)
        assert v1['success'], v1['reason']

        now += timedelta(seconds=0.9)

        v2 = reserve(now, 'entry_w', 'exit_e', max_window=1, debug=True)
        assert v2['success'], v2['reason']

        diff = (v2['time_ref'] - v1['time_ref']).total_seconds()
        n = int(np.ceil(diff / 0.2))
        timeline = new_timeline(0.2*n + 5)
        v1_part = v1['analysis']['pass4']
        v2_part = v2['analysis']['pass4']
        v1_full = np.ones(timeline.shape + v1_part.shape[1:])
        v2_full = np.ones(timeline.shape + v2_part.shape[1:])
        v1_full[0:0+26] = v1_part
        v2_full[n:n+26] = v2_part

        v2_depart   = np.ones(timeline.shape + v2_part.shape[1:])
        v2_arrival  = np.ones(timeline.shape + v2_part.shape[1:])
        v2_depart[n:n+26] = v2['analysis']['depart_target']
        v2_arrival[n:n+26] = v2['analysis']['arrival_target']

        v2_pass2 = np.ones(timeline.shape + v2_part.shape[1:])
        v2_pass2[n:n+26] = v2['analysis']['pass2']

        def keep(vf, ax, lo, hi):
            out = np.ones_like(vf)
            idx = tuple(slice(lo, hi) if i == ax else slice(0, n)
                        for i, n in enumerate(vf.shape))
            out[idx] = vf[idx]
            return out

        interact_tubes(timeline, 
                       ('reds', keep(v1_full, 2, 9, 12)), 
                       ('greys', v2_pass2, dict(opacity=0.4)),
                       ('blues', v2_full), 
                       ('greens', v2_depart),
                       ('greens', v2_arrival),
                       bgpath='../data/4way.png')().write_html('plot.html')
        
        new_map(
            (shp.project_onto(v1_full, 0, 1, 2)[14], dict(cmap='Reds')), 
            (shp.project_onto(v2_full, 0,1,2)[14], dict(cmap='Blues')),
            bgpath='../data/4way.png'
        ).savefig('plot-xy.png')

        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(9*4/3, 9))
        ax.set_xlabel('x [m]')
        ax.set_ylabel('t [s]')
        plot_levelset_many(
            (shp.project_onto(v1_full[:,:,9:12], 0, 1), dict(cmap='Reds')),
            (shp.project_onto(v2_pass2[:,:,9:12], 0, 1), dict(cmap='Greys', alpha=0.3)),
            (shp.project_onto(v2_full[:,:,9:12], 0, 1), dict(cmap='Blues')),
            (shp.project_onto(v2_depart[:,:,9:12], 0, 1), dict(cmap='Greens')),
            (shp.project_onto(v2_arrival[:,:,9:12], 0, 1), dict(cmap='Greens')),
            ax=ax, 
            alpha=0.9, 
            aspect=2.5/timeline.max(), 
            extent=[-1.25, +1.25, 0, timeline.max()],
        )
        fig.tight_layout()
        fig.savefig('plot-xt.png')

        # x=-0.75, y=-0.25, h=0, v=?
        ix = val2ind(-0.5, 0)
        iy = val2ind(-0.25, 1)
        ih = val2ind(0, 2)
        # [13, ix, iy, ih, :]
        vf = v2_full[14]
        dt = 0.2
        shape = 31, 31, 25, 7
        x, y, h, v = np.meshgrid(*[np.linspace(REG['min_bounds'][i], REG['max_bounds'][i], shape[i])
                                   for i in range(4)])
        f = np.array([
            v * np.cos(h),
            v * np.sin(h),
            np.zeros_like(h),
            np.zeros_like(v),
        ])
        g = np.array([
            [0],
            [0],
            [0],
            [1],
        ])
        # idx = np.where(vf <= 0)
        idx = (ix, iy, ih, slice(0, 7))
        dvdx = np.array(np.gradient(vf))[(...,) + idx]

        a = vf[idx] + dt*np.sum(dvdx * f[(...,) + idx], axis=0)
        b = (g.T @ dvdx)
        u = np.linspace(-0.4, 0.4, 25).reshape(len(b), -1)
        lrcs = a.reshape(-1, 1) + b.T @ u

        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(9*4/3, 9))
        ax.set_xlabel(r'Velocity [$m/s$]')
        ax.set_ylabel(r'Acceleration [$m/s^2$]')
        ax.set_ylim([-0.5, +0.5])
        plot_levelset(lrcs, 
                      ax=ax,
                      transpose=True,
                      cmap='Blues', cval=0.75,
                      aspect=(REG['max_bounds'][3] - REG['min_bounds'][3]),
                      extent=[REG['min_bounds'][3],
                              REG['max_bounds'][3],
                              -0.4, +0.4])
        fig.tight_layout()
        fig.savefig('plot-lrcs.png')

    EYE_W   = sphere_to_cartesian(2.2, 45, -90 - 90)
    EYE_WSW = sphere_to_cartesian(2.2, 70, -90 - 70)
    EYE_SW  = sphere_to_cartesian(2.5, 60, -90 - 45)
    EYE_SSW = sphere_to_cartesian(2.2, 70, -90 - 20)
    EYE_S   = sphere_to_cartesian(2.5, 45, -90 + 0)
    EYE_SSE = sphere_to_cartesian(2.2, 70, -90 + 20)
    EYE_SE  = sphere_to_cartesian(2.5, 60, -90 + 45)
    EYE_ESE = sphere_to_cartesian(2.2, 70, -90 + 70)
    EYE_E   = sphere_to_cartesian(2.2, 45, -90 + 90)