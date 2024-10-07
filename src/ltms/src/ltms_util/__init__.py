from math import ceil
from time import time, sleep

import hj_reachability as hj

from .rc import RC
from .env import create_4way
from .solver import *

from .bob import Bob

import sys
if sys.version_info.minor >= 10:

    from .plotting import *

    load_plot = lambda name, *args: new_map(shp.project_onto(np.load(name), *args)).show()

    
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

    def time_test():
        from datetime import datetime, timedelta

        bob = Bob(interactive=False)

        vehicles = [
            ('v1', 0.0, 'entry_e', 'exit_n'),
            ('v2', 1.6, 'entry_e', 'exit_n'),
            ('v3', 3.2, 'entry_e', 'exit_n'),
        ]
        time_table = {name: [] for name, *_ in vehicles}

        for _ in range(5):
            now = datetime(1, 1, 1)

            for name, delta, entry_loc, exit_loc in vehicles:
                r = bob.reserve(now + timedelta(seconds=delta), entry_loc, exit_loc, max_window=1.5)
                assert r['success'], r['reason']
                time_table[name].append(r['analysis']['comp_time'])

            bob.clean_all()
            bob._reservations = {}
            print(', '.join(f'{name}={times[-1]:.02f}' for name, times in time_table.items()))
            sleep(1)

        avg = lambda seq: sum(seq)/len(seq)
        for name, times in time_table.items():
            print(f'{name}: avg={avg(times):.02f}, max={max(times):.02f}, min={min(times):.02f}')


    def figs():
        from datetime import datetime, timedelta
        import matplotlib as mpl

        mpl.rcParams.update(**{'font.size': 22})

        bob = Bob(interactive=True)

        max_window = 1.5

        # now = datetime.now()
        now = datetime(1, 1, 1)

        v1, v2, v3 = bob.reserve_many(dict(time_ref=now + timedelta(seconds=0.0),
                                           entry_loc='entry_e', exit_loc='exit_n', 
                                           max_window=max_window,
                                           debug=True),
                                      dict(time_ref=now + timedelta(seconds=1.5), 
                                           entry_loc='entry_e', exit_loc='exit_n', 
                                           max_window=max_window,
                                           debug=True),
                                      dict(time_ref=now + timedelta(seconds=3.0), 
                                           entry_loc='entry_e', exit_loc='exit_n', 
                                           max_window=max_window,
                                           debug=True))

        
        first = min(v1['time_ref'], v2['time_ref'], v3['time_ref'])
        last = max(v1['time_ref'], v2['time_ref'], v3['time_ref'])
        n = int(np.ceil((last-first).total_seconds() / 0.2))
        timeline = new_timeline(0.2*n + 5)
        v1_part = v1['analysis']['pass4']
        v2_part = v2['analysis']['pass4']
        v3_part = v3['analysis']['pass4']

        v1_full = np.ones(timeline.shape + v1_part.shape[1:])
        v2_full = np.ones(timeline.shape + v2_part.shape[1:])
        v3_full = np.ones(timeline.shape + v3_part.shape[1:])

        diff = (v1['time_ref'] - first).total_seconds()
        n = int(np.ceil(diff / 0.2))
        v1_full[n:n+26] = v1_part
        
        diff = (v2['time_ref'] - first).total_seconds()
        n = int(np.ceil(diff / 0.2))
        v2_full[n:n+26] = v2_part
        
        diff = (v3['time_ref'] - first).total_seconds()
        n = int(np.ceil(diff / 0.2))
        v3_full[n:n+26] = v3_part

        v2_depart   = np.ones(timeline.shape + v2_part.shape[1:])
        v2_arrival  = np.ones(timeline.shape + v2_part.shape[1:])
        v2_depart[n:n+26] = v2['analysis']['entry_target']
        v2_arrival[n:n+26] = v2['analysis']['exit_target']

        v2_pass2 = np.ones(timeline.shape + v2_part.shape[1:])
        v2_pass2[n:n+26] = v2['analysis']['pass2']

        def keep(vf, ax, lo, hi):
            out = np.ones_like(vf)
            idx = tuple(slice(lo, hi) if i == ax else slice(0, n)
                        for i, n in enumerate(vf.shape))
            out[idx] = vf[idx]
            return out

        interact_tubes(timeline, 
                       ('reds', v1_full), 
                       ('blues', v2_full), 
                       ('purples', v3_full), 
                       bgpath='../data/4way.png')().write_html('plot.html')

        interact_tubes(timeline, 
                       ('reds', keep(v1_full, 2, 9, 12)), 
                       ('greys', v2_pass2, dict(opacity=0.4)),
                       ('blues', v2_full), 
                       ('greens', v2_depart),
                       ('greens', v2_arrival),
                       bgpath='../data/4way.png')().write_html('plot-slice.html')
        
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

        # # x=-0.75, y=-0.25, h=0, v=?
        # ix = val2ind(-0.5, 0)
        # iy = val2ind(-0.25, 1)
        # ih = val2ind(0, 2)
        # # [13, ix, iy, ih, :]
        #vf = v2_full[14]
        dt = 0.2
        shape = 31, 31, 25, 7
        x, y, h, v = np.meshgrid(*[np.linspace(RC['min_bounds'][i], RC['max_bounds'][i], shape[i])
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

        # # idx = np.where(vf <= 0)
        # idx = (ix, iy, ih, slice(0, 7))
        # dvdx = np.array(np.gradient(vf))[(...,) + idx]

        # a = vf[idx] + dt*np.sum(dvdx * f[(...,) + idx], axis=0)
        # b = (g.T @ dvdx)
        # u = np.linspace(-0.4, 0.4, 25).reshape(len(b), -1)
        # lrcs = a.reshape(-1, 1) + b.T @ u

        for vf_full in (v1_full, v2_full, v3_full):

            times = [] 
            arr = np.random.rand(100, 4)
            arr[:, 0] *= len(timeline) - 1
            arr[:, 1:3] *= 31 - 1
            arr[:, 3] *= 25 - 1
            arr = np.round(arr).astype(int)

            for it, ix, iy, ih in arr:
                start = time()
                # # x=-0.75, y=-0.25, h=0, v=?
                # ix = val2ind(-0.5, 0)
                # iy = val2ind(-0.25, 1)
                # ih = val2ind(0, 2)
                # [13, ix, iy, ih, :]
                # vf = v2_full[14]

                vf = vf_full[it]
                # idx = np.where(vf <= 0)
                idx = (ix, iy, ih, slice(0, 7))
                dvdx = np.array(np.gradient(vf))[(...,) + idx]

                a = vf[idx] + dt*np.sum(dvdx * f[(...,) + idx], axis=0)
                b = (g.T @ dvdx)
                u = np.linspace(-0.4, 0.4, 25).reshape(len(b), -1)
                lrcs = a.reshape(-1, 1) + b.T @ u
                stop = time()

                times.append(stop-start)
            print(sum(times)/len(times))


        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(9*4/3, 9))
        ax.set_xlabel(r'Velocity [$m/s$]')
        ax.set_ylabel(r'Acceleration [$m/s^2$]')
        ax.set_ylim([-0.5, +0.5])
        plot_levelset(lrcs, 
                      ax=ax,
                      transpose=True,
                      cmap='Blues', cval=0.75,
                      aspect=(RC['max_bounds'][3] - RC['min_bounds'][3]),
                      extent=[RC['min_bounds'][3],
                              RC['max_bounds'][3],
                              -0.4, +0.4])
        fig.tight_layout()
        fig.savefig('plot-lrcs.png')

    def chaos():

        def around(l, e, n):
            if e not in l: return []
            i = l.index(e)
            N = len(l)
            return [l[(i+j) % N] for j in range(-n, n+1)]

        RC['model'] = 'Bicycle5D'
        RC['min_bounds'] = [-1.5, -1.5, -np.pi, -np.pi/5, +0.0]
        RC['max_bounds'] = [+1.5, +1.5, +np.pi, +np.pi/5, +0.6]
        RC['grid_shape'] = (31, 31, 25, 5, 7)

        RC['extent'] = [RC['min_bounds'][0], RC['max_bounds'][0], 
                        RC['min_bounds'][1], RC['max_bounds'][1]]

        RC['locations'] = [
            'center_e', 'center_ene', 'center_ne', 'center_nne',
            'center_n', 'center_nnw', 'center_nw', 'center_wnw',
            'center_w', 'center_wsw', 'center_sw', 'center_ssw',
            'center_s', 'center_ese', 'center_se', 'center_sse',
        ]
        RC['permitted_routes'] = {
            (_entry, _exit): ('outside',)
            for _entry in RC['locations']
            for _exit in set(RC['locations']) - set(around(RC['locations'], _entry, 4)) # flip
        }

        RC['entry_locations']   = RC['locations'] + ['init']
        RC['exit_locations']    = RC['locations']
        RC['locations'] += ['outside']
        RC['permitted_routes'].update({
            ('init', _exit): ('outside',)
            for _exit in 'center_ne center_ene center_e'.split()
        })

        RC['bob']['model']              = RC['model']
        RC['bob']['grid_shape']         = RC['grid_shape']
        RC['bob']['min_bounds']         = RC['min_bounds']
        RC['bob']['max_bounds']         = RC['max_bounds']

        from datetime import datetime, timedelta
        import matplotlib as mpl

        mpl.rcParams.update(**{'font.size': 22})

        bob = Bob(interactive=True)

        max_window = 1.5

        # now = datetime.now()
        now = datetime(1, 1, 1)

        v1, v2 = bob.reserve_many(dict(time_ref=now + timedelta(seconds=0.0),
                                       entry_loc='init', exit_loc='center_e', 
                                       max_window=max_window,
                                       debug=True),
                                  dict(time_ref=now + timedelta(seconds=1.5), 
                                       entry_loc='init', exit_loc='center_e', 
                                       max_window=max_window,
                                       debug=True))

        
        first = min(v1['time_ref'], v2['time_ref'])
        last = max(v1['time_ref'], v2['time_ref'])
        n = int(np.ceil((last-first).total_seconds() / 0.2))
        timeline = new_timeline(0.2*n + 5)
        v1_part = v1['analysis']['pass4']
        v2_part = v2['analysis']['pass4']

        v1_full = np.ones(timeline.shape + v1_part.shape[1:])
        v2_full = np.ones(timeline.shape + v2_part.shape[1:])

        diff = (v1['time_ref'] - first).total_seconds()
        n = int(np.ceil(diff / 0.2))
        v1_full[n:n+26] = v1_part
        
        diff = (v2['time_ref'] - first).total_seconds()
        n = int(np.ceil(diff / 0.2))
        v2_full[n:n+26] = v2_part
        
        v2_depart   = np.ones(timeline.shape + v2_part.shape[1:])
        v2_arrival  = np.ones(timeline.shape + v2_part.shape[1:])
        v2_depart[n:n+26] = v2['analysis']['entry_target']
        v2_arrival[n:n+26] = v2['analysis']['exit_target']

        v2_pass2 = np.ones(timeline.shape + v2_part.shape[1:])
        v2_pass2[n:n+26] = v2['analysis']['pass2']

        def keep(vf, ax, lo, hi):
            out = np.ones_like(vf)
            idx = tuple(slice(lo, hi) if i == ax else slice(0, n)
                        for i, n in enumerate(vf.shape))
            out[idx] = vf[idx]
            return out

        interact_tubes(timeline, 
                       ('reds', v1_full), 
                       ('blues', v2_full), 
                       bgpath='../data/4way.png')().write_html('plot.html')

        interact_tubes(timeline, 
                       ('reds', keep(v1_full, 2, 9, 12)), 
                       ('greys', v2_pass2, dict(opacity=0.4)),
                       ('blues', v2_full), 
                       ('greens', v2_depart),
                       ('greens', v2_arrival),
                       bgpath='../data/4way.png')().write_html('plot-slice.html')
        
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
