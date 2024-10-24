from math import ceil
from time import time
from struct import pack

import numpy as np
import numpy.lib.stride_tricks as st

import hj_reachability as hj
import hj_reachability.shapes as shp

from .rc import RC
from .util import iterflat, setdefaults

def trybreak(f):
    def wrapper(*args, **kwds):
        try:
            return f(*args, **kwds)
        except Exception:
            breakpoint()
            return f(*args, **kwds)
    return wrapper
        

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

import numpy as np

def latest_window(mask, N=1, M=None):
    """
    Find the last maximal window where at least N but less than M consecutive elements are True.
    
    Parameters:
    mask (array-like): A boolean array.
    N (int): The minimum length of consecutive True values.
    M (int or None): The maximum length of consecutive True values (None means no upper limit).
    
    Returns:
    np.ndarray: Indices of the last maximal window of True values.
    """
    mask = np.asarray(mask)
    length = len(mask)
    
    # Variables to track the current window
    current_start = None
    current_length = 0
    
    # Variables to track the best (latest) window that satisfies the constraints
    best_start = None
    best_end = None
    
    # Traverse through the array to find connected windows of True values
    for i in range(length):
        if mask[i]:
            if current_start is None:
                current_start = i  # Start a new window
            current_length += 1
        else:
            # Check if the current window satisfies the N and M constraints
            if current_length >= N and (M is None or current_length < M):
                best_start = current_start
                best_end = i  # End of the window is the last False element
            # Reset the current window
            current_start = None
            current_length = 0
    
    # Final check at the end of the array (in case the array ends with a valid window)
    if current_length >= N and (M is None or current_length < M):
        best_start = current_start
        best_end = length  # End is the end of the array
    
    # If we found a valid window, return its indices
    if best_start is not None and best_end is not None:
        return np.arange(best_start, best_end)
    
    # If no valid window was found, return an empty array
    return np.array([], int)


def val2ind(x, axis, **kwargs):
    setdefaults(kwargs,
                min_bounds=RC['min_bounds'],
                max_bounds=RC['max_bounds'],
                grid_shape=RC['grid_shape'])
    min_bounds = kwargs.pop('min_bounds')
    max_bounds = kwargs.pop('max_bounds')
    grid_shape = kwargs.pop('grid_shape')

    dx = (max_bounds[axis] - min_bounds[axis]) / grid_shape[axis]
    return round((x - min_bounds[axis]) / dx) - 1

def ind2val(i, axis, **kwargs):
    setdefaults(kwargs,
                min_bounds=RC['min_bounds'],
                max_bounds=RC['max_bounds'],
                grid_shape=RC['grid_shape'])
    min_bounds = kwargs.pop('min_bounds')
    max_bounds = kwargs.pop('max_bounds')
    grid_shape = kwargs.pop('grid_shape')

    dx = (max_bounds[axis] - min_bounds[axis]) / grid_shape[axis]
    return i*dx + min_bounds[axis]


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
                  *iterflat([[self.grid.domain.lo[i], self.grid.domain.hi[i], self.grid.shape[i]]
                              for i in range(self.grid.ndim)]))
        self.code_grid = bytes([sum(bs) % 256]).hex()

        bs = bytes()
        bs += pack('sx', cls.__name__.encode())
        for key, val in dynamics.items():
            bs += pack('sxf', key.encode(), val)
        self.code_dynamics = bytes([sum(bs) % 256]).hex()
    
    def nearest_index(self, x):
        x = np.array(x)
        # assert (x >= min_bounds).all() and (x <= max_bounds).all(), f'Point {x} is out of bounds'
        ix = np.array(self.grid.nearest_index(x), int)
        ix = np.where(ix >= self.grid.shape, np.array(self.grid.shape)-1, ix)
        return tuple(ix)
    
    def spatial_deriv(self, vf, ix):
        spatial_deriv = []

        for axis in range(len(ix)):
            
            ix_nxt = list(ix)
            ix_nxt[axis] += 1
            ix_nxt = tuple(ix_nxt)

            ix_prv = list(ix)
            ix_prv[axis] -= 1
            ix_prv = tuple(ix_prv)

            sign = np.sign(vf[ix])

            if ix[axis] == 0:
                leftv = (vf[ix_nxt[:axis] + (-1,) + ix_nxt[axis+1:]] if self.grid._is_periodic_dim[axis] else 
                        vf[ix] + sign*np.abs(vf[ix_nxt] - vf[ix]))
                rightv = vf[ix_nxt]
            elif ix[axis] == self.grid.shape[axis] - 1:
                leftv = vf[ix_prv]
                rightv = (vf[ix_prv[:axis] + (0,) + ix_prv[axis+1:]] if self.grid._is_periodic_dim[axis] else 
                        vf[ix] + sign*np.abs(vf[ix] - vf[ix_prv]))
            else:
                leftv = vf[ix_prv]
                rightv = vf[ix_nxt]

            left_dx = (vf[ix] - leftv) / self.grid.spacings[axis]
            right_dx = (rightv - vf[ix]) / self.grid.spacings[axis]
            spatial_deriv.append((left_dx + right_dx) / 2)

        return np.array(spatial_deriv)

    # @trybreak
    def brs(self, times, target, constraints=None, *, mode='reach', interactive=True):
        jnp = hj.solver.jnp
        times = -jnp.asarray(times)
        target = jnp.asarray(target)
        if constraints is not None:
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
                          progress_bar=False) # progress_bar=interactive)
        values = jnp.flip(values, axis=0)
        return np.asarray(values)

    # @trybreak
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
                          progress_bar=False) # progress_bar=interactive)
        return np.asarray(values)
    
    def lrcs(self, vf, x, i):
        f = self.reach_dynamics.open_loop_dynamics(x, 0)
        g = self.reach_dynamics.control_jacobian(x, 0)

        ix = self.nearest_index(x)
        dvdx = self.spatial_deriv(vf[i], ix)

        a = np.array(vf[(i+1, *ix)] + self.time_step*(dvdx.T @ f))
        b = np.array(self.time_step*(dvdx.T @ g))

        control_space = np.array([
            self.reach_dynamics.control_space.lo,
            self.reach_dynamics.control_space.hi,
        ]).T
        control_vecs = [np.linspace(*lohi) for lohi in control_space]
        control_grid = np.meshgrid(*control_vecs)

        # This is the important part.
        # Essentially we want: a + b \cdot u <= 0
        # Here, `mask` masks the set over the control space spanned by `control_vecs`
        terms = [us*b_ for us, b_ in zip(control_grid, b)]
        mask = sum(terms, a) <= 0
        return mask, control_vecs

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
        start_time_all = time()

        debug = kwargs.pop('debug', False)
        interactive = kwargs.pop('interactive', self.is_interactive)

        min_window = kwargs.pop('min_window', 1)  # Need at least 1 second to enter/exit
        max_window = kwargs.pop('max_window', 2)  # Allow up to 2 seconds to enter/exit
        min_window_entry = kwargs.pop('min_window_entry', min_window)
        max_window_entry = kwargs.pop('max_window_entry', max_window)
        min_window_exit = kwargs.pop('min_window_exit', min_window)
        max_window_exit = kwargs.pop('max_window_exit', max_window)
        result = kwargs.pop('result', {})

        ALL_PASSES = ['pass1', 'pass2', 'pass3', 'pass4']
        passes = passes or ALL_PASSES
        if 'all' in passes:
            passes = ALL_PASSES
        passes = [name for name in ALL_PASSES if name in passes]

        passes_out = ALL_PASSES if debug else list(passes)

        for name in passes:
            assert name in ALL_PASSES, f'Invalid pass: {name}'
            i = ALL_PASSES.index(name)
            
            if i > 0 and ALL_PASSES[i-1] not in passes + list(kwargs):
                passes += [ALL_PASSES[i-1]]
        passes_sch = [name for name in ALL_PASSES if name in passes]

        if interactive:
            msg = 'Running analysis with the following passes:\n'
            msg += '  Scheduled: ' + ', '.join(passes_sch) + '\n'
            msg += '  resulting: ' + ', '.join(passes_out) + '\n'
            underline = '-' * max(map(len, msg.split('\n')))
            print(msg + underline, end='\n\n')

        def to_shared(values, **kwargs):
            return shp.project_onto(values, 0, 1, 2, **kwargs)
        def when_overlapping(a, b):
            return np.where(shp.project_onto(shp.intersection(a, b), 0) <= 0)[0]
        
        if 'pass1' in passes_sch:
            if interactive: print('Pass 1: Initial BRS')
            
            rules = kwargs['constraints']
            end = kwargs['exit']
            comp_time = 0

            target = shp.make_tube(self.timeline, end)
            constraints = rules

            start_time = time()
            vf = self.brs(self.timeline, target, constraints, interactive=interactive)
            stop_time = time()
            comp_time = stop_time - start_time

            if interactive:
                print(f'  Compute Time: {comp_time:.02f}')
            
            kwargs['pass1'] = vf

            result['comp_time_pass1'] = comp_time
            if 'pass1' in passes_out: result['pass1'] = vf.copy()

        if 'pass2' in passes_sch:
            if interactive: print('Pass 2: Avoidance')
            
            vf = kwargs['pass1']
            avoid = kwargs['dangers']
            end = kwargs['exit']
            comp_time = 0

            if avoid:
                avoid_target = shp.union(
                    *map(lambda target: to_shared(target, keepdims=True), avoid),
                ) - self.AVOID_MARGIN # increase avoid set with heuristic margin

                interact_window = when_overlapping(vf, avoid_target)
                if 0 < interact_window.size:
                    i, j = interact_window[0], interact_window[-1] + 1

                    # Recompute solution until after last interaction
                    constraints = shp.setminus(vf, avoid_target)[:j+1]
                    target = vf[:j+1]    # last step is target to reach pass1
                    target[:-1] = end       # all other steps are recomputed to end
                    
                    start_time = time()
                    vf[:j+1] = self.brs(self.timeline[:j+1], target, constraints, interactive=interactive)
                    stop_time = time()
                    comp_time = stop_time - start_time
                
                    if interactive:
                        print(f'  First Interaction: {self.timeline[i]:.01f}')
                        print(f'  Last Interaction: {self.timeline[j-1]:.01f}')
                        print(f'  Compute Time: {comp_time:.02f}')

            kwargs['pass2'] = vf

            result['comp_time_pass2'] = comp_time
            if 'pass2' in passes_out: result['pass2'] = vf.copy()

        if 'pass3' in passes_sch:
            if interactive: print('Pass 3: Planning, Entry')
            
            vf = kwargs['pass2']
            start = kwargs['entry']
            comp_time = 0

            min_nsteps = ceil(min_window_entry / self.time_step)
            max_nsteps = ceil(max_window_entry / self.time_step)
            depart_target = shp.intersection(vf, start)
            depart_window = earliest_window(shp.project_onto(depart_target, 0) <= 0, min_nsteps)
            assert depart_window.size > 0, 'Analysis Failed: No time window to enter region'
            w0 = 0
            wn = min(max_nsteps+1, len(depart_window)-1)
            i = depart_window[w0] # Earliest entry index
            j = depart_window[wn] # Latest entry index
            depart_target[j:] = 1

            start_time = time()
            vf[i:] = self.frs(self.timeline[i:], depart_target[i:], vf[i:], interactive=interactive)
            vf[:i] = 1 # Remove all values before departure
            stop_time = time()
            comp_time = stop_time - start_time

            if interactive:
                print(f'  Earliest Entry: {self.timeline[i]:.01f}')
                print(f'  Latest Entry: {self.timeline[j-1]:.01f}')
                print(f'  Compute Time: {comp_time:.02f}')

            kwargs['pass3'] = vf
            
            result['comp_time_pass3'] = comp_time
            result['earliest_entry'] = self.timeline[i]
            result['latest_entry'] = self.timeline[j-1]
            if 'pass3' in passes_out: result['pass3'] = vf.copy()

            if debug: result['entry_target'] = depart_target

        if 'pass4' in passes_sch:
            if interactive: print('Pass 4: Planning, Exit')
            
            vf = kwargs['pass3']
            end = kwargs['exit']
            comp_time = 0

            min_nsteps = ceil(min_window_exit / self.time_step)
            max_nsteps = ceil(max_window_exit / self.time_step)
            arrival_target = shp.intersection(vf, end)
            arrival_target_times = shp.project_onto(arrival_target, 0, 1, 2)
            arrival_target_times = shp.project_onto(arrival_target_times, 0, union=False)
            arrival_window = earliest_window(arrival_target_times <= 0, min_nsteps)
            print(arrival_target_times <= 0, arrival_window,  j)
            assert arrival_window.size > 0, 'Analysis Failed: No time window to exit region'
            w0 = 0
            wn = min(max_nsteps+1, len(arrival_window)-1)
            m = arrival_window[w0] # Earliest exit index
            n = arrival_window[wn] # Latest exit index
            arrival_target[n:] = 1

            start_time = time()
            vf[i:n] = self.brs(self.timeline[i:n], arrival_target[i:n], vf[i:n], interactive=interactive)
            vf[n:] = 1 # Remove all values after arrival
            stop_time = time()
            comp_time = stop_time - start_time

            if interactive:
                print(f'  Earliest Exit: {self.timeline[m]:.01f}')
                print(f'  Latest Exit: {self.timeline[n-1]:.01f}')
                print(f'  Compute Time: {comp_time:.02f}')

            kwargs['pass4'] = vf

            result['comp_time_pass4'] = comp_time
            result['earliest_exit'] = self.timeline[m]
            result['latest_exit'] = self.timeline[n-1]
            if 'pass4' in passes_out: result['pass4'] = vf.copy()

            if debug: result['exit_target'] = arrival_target

        stop_time_all = time()
        comp_time = stop_time_all - start_time_all

        if interactive:
            print(f'Total Compute Time: {comp_time:.02f}')
            print(underline, end='\n\n')

        result['comp_time'] = comp_time

        return result

    def run_many(self, *objectives, **kwargs):
        interactive = kwargs.get('interactive', self.is_interactive)
        results = []
        for o in objectives:
            results += self.run_analysis('pass4', **o, **kwargs, avoid=results)
            if interactive: print('\n')
        return results
