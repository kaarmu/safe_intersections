from math import pi

import hj_reachability.shapes as shp


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
