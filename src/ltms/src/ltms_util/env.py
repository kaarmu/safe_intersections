from math import pi

import hj_reachability.shapes as shp

def create_chaos(grid, *envs):
    out = {}

    if grid.ndim == 5:
        X, Y, H, D, V = range(grid.ndim)
    elif grid.ndim == 4:
        X, Y, H, V = range(grid.ndim)

    X0 = grid.domain.lo[X]
    Y0 = grid.domain.lo[Y]

    XN = grid.domain.hi[X] - grid.domain.lo[X]
    YN = grid.domain.hi[Y] - grid.domain.lo[Y]

    centers = ('center_e', 'center_ene', 'center_ne', 'center_nne',
               'center_n', 'center_nnw', 'center_nw', 'center_wnw',
               'center_w', 'center_wsw', 'center_sw', 'center_ssw',
               'center_s', 'center_ese', 'center_se', 'center_sse')

    if not envs:
        envs += centers
        envs += ('init', 'full')

    envs_out = tuple(envs)
    envs_sch = tuple(envs)

    if set(centers) & set(envs_sch) and 'center' not in envs_sch:
        envs_sch += ('center',)
    if 'full' in envs_sch and 'init' not in envs_sch:
        envs_sch += ('init',)

    (H_W, H_WSW, H_SW, H_SSW,
     H_S, H_ESE, H_SE, H_SSE,
     H_E, H_ENE, H_NE, H_NNE,
     H_N, H_NNW, H_NW, H_WNW) = [-pi + (2*pi/16)*x for x in range(16)]

    H_MARGIN = 2*pi/16 * 0.8

    if 'center' in envs_sch:
        out['center'] = shp.rectangle(grid, axes=[X, Y],
                                      target_min=[X0 + (0.5-0.1)*XN, Y0 + (0.5-0.1)*YN],
                                      target_max=[X0 + (0.5+0.1)*XN, Y0 + (0.5+0.1)*YN])

    if 'center_w' in envs_sch:
        out['center_w'] = shp.intersection(
            shp.rectangle(grid, axes=[H],
                          target_min=[H_W - H_MARGIN],
                          target_max=[H_W + H_MARGIN]),
            out['center'],
        )
    if 'center_wsw' in envs_sch:
        out['center_wsw'] = shp.intersection(
            shp.rectangle(grid, axes=[H],
                          target_min=[H_WSW - H_MARGIN],
                          target_max=[H_WSW + H_MARGIN]),
            out['center'],
        )
    if 'center_sw' in envs_sch:
        out['center_sw'] = shp.intersection(
            shp.rectangle(grid, axes=[H],
                          target_min=[H_SW - H_MARGIN],
                          target_max=[H_SW + H_MARGIN]),
            out['center'],
        )
    if 'center_ssw' in envs_sch:
        out['center_ssw'] = shp.intersection(
            shp.rectangle(grid, axes=[H],
                          target_min=[H_SSW - H_MARGIN],
                          target_max=[H_SSW + H_MARGIN]),
            out['center'],
        )
    if 'center_s' in envs_sch:
        out['center_s'] = shp.intersection(
            shp.rectangle(grid, axes=[H],
                          target_min=[H_S - H_MARGIN],
                          target_max=[H_S + H_MARGIN]),
            out['center'],
        )
    if 'center_ese' in envs_sch:
        out['center_ese'] = shp.intersection(
            shp.rectangle(grid, axes=[H],
                          target_min=[H_ESE - H_MARGIN],
                          target_max=[H_ESE + H_MARGIN]),
            out['center'],
        )
    if 'center_se' in envs_sch:
        out['center_se'] = shp.intersection(
            shp.rectangle(grid, axes=[H],
                          target_min=[H_SE - H_MARGIN],
                          target_max=[H_SE + H_MARGIN]),
            out['center'],
        )
    if 'center_sse' in envs_sch:
        out['center_sse'] = shp.intersection(
            shp.rectangle(grid, axes=[H],
                          target_min=[H_SSE - H_MARGIN],
                          target_max=[H_SSE + H_MARGIN]),
            out['center'],
        )
    if 'center_e' in envs_sch:
        out['center_e'] = shp.intersection(
            shp.rectangle(grid, axes=[H],
                          target_min=[H_E - H_MARGIN],
                          target_max=[H_E + H_MARGIN]),
            out['center'],
        )
    if 'center_ene' in envs_sch:
        out['center_ene'] = shp.intersection(
            shp.rectangle(grid, axes=[H],
                          target_min=[H_ENE - H_MARGIN],
                          target_max=[H_ENE + H_MARGIN]),
            out['center'],
        )
    if 'center_ne' in envs_sch:
        out['center_ne'] = shp.intersection(
            shp.rectangle(grid, axes=[H],
                          target_min=[H_NE - H_MARGIN],
                          target_max=[H_NE + H_MARGIN]),
            out['center'],
        )
    if 'center_nne' in envs_sch:
        out['center_nne'] = shp.intersection(
            shp.rectangle(grid, axes=[H],
                          target_min=[H_NNE - H_MARGIN],
                          target_max=[H_NNE + H_MARGIN]),
            out['center'],
        )
    if 'center_n' in envs_sch:
        out['center_n'] = shp.intersection(
            shp.rectangle(grid, axes=[H],
                          target_min=[H_N - H_MARGIN],
                          target_max=[H_N + H_MARGIN]),
            out['center'],
        )
    if 'center_nnw' in envs_sch:
        out['center_nnw'] = shp.intersection(
            shp.rectangle(grid, axes=[H],
                          target_min=[H_NNW - H_MARGIN],
                          target_max=[H_NNW + H_MARGIN]),
            out['center'],
        )
    if 'center_nw' in envs_sch:
        out['center_nw'] = shp.intersection(
            shp.rectangle(grid, axes=[H],
                          target_min=[H_NW - H_MARGIN],
                          target_max=[H_NW + H_MARGIN]),
            out['center'],
        )
    if 'center_wnw' in envs_sch:
        out['center_wnw'] = shp.intersection(
            shp.rectangle(grid, axes=[H],
                          target_min=[H_WNW - H_MARGIN],
                          target_max=[H_WNW + H_MARGIN]),
            out['center'],
        )

    if 'init' in envs_sch:
        out['init'] = shp.rectangle(grid, axes=[X, Y, H],
                                    target_min=[X0,            Y0,              H_N - H_MARGIN],
                                    target_max=[X0 + 0.15*XN, Y0 + 0.23*YN,     H_N + H_MARGIN])
    if 'full' in envs_sch:
        out['full'] = shp.rectangle(grid, axes=[V],
                                    target_min=[0.3],
                                    target_max=[0.6])

    return {name: out[name] for name in envs_out}

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
                                    target_min=[Y0 + 0.3*YN, -pi/5],
                                    target_max=[Y0 + 0.5*YN, +pi/5])
        out['road_e'] = shp.union(shp.intersection(out['road_e'], speedlimit), out['center'])
    if 'road_w' in envs_sch:
        out['road_w'] = shp.rectangle(grid,
                                    axes=[Y, A],
                                    target_min=[Y0 + 0.5*YN, +pi - pi/5],
                                    target_max=[Y0 + 0.7*YN, -pi + pi/5])
        out['road_w'] = shp.union(shp.intersection(out['road_w'], speedlimit), out['center'])
    if 'road_n' in envs_sch:
        out['road_n'] = shp.rectangle(grid,
                                    axes=[X, A],
                                    target_min=[X0 + 0.5*XN, +pi/2 - pi/5],
                                    target_max=[X0 + 0.7*XN, +pi/2 + pi/5])
        out['road_n'] = shp.union(shp.intersection(out['road_n'], speedlimit), out['center'])
    if 'road_s' in envs_sch:
        out['road_s'] = shp.rectangle(grid,
                                    axes=[X, A],
                                    target_min=[X0 + 0.3*XN, -pi/2 - pi/5],
                                    target_max=[X0 + 0.5*XN, -pi/2 + pi/5])
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
