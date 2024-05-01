from functools import wraps

import numpy as np
import skimage.io as sio
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import hj_reachability.shapes as shp

from .util import setdefaults
from .rc import RC



def sphere_to_cartesian(r, theta, phi):
    theta *= np.pi/180
    phi *= np.pi/180
    return dict(x=r*np.sin(theta)*np.cos(phi),
                y=r*np.sin(theta)*np.sin(phi),
                z=r*np.cos(theta))

EYE_W   = sphere_to_cartesian(2.2, 45, -90 - 90)
EYE_WSW = sphere_to_cartesian(2.2, 70, -90 - 70)
EYE_SW  = sphere_to_cartesian(2.5, 60, -90 - 45)
EYE_SSW = sphere_to_cartesian(2.2, 70, -90 - 20)
EYE_S   = sphere_to_cartesian(2.5, 45, -90 + 0)
EYE_SSE = sphere_to_cartesian(2.2, 70, -90 + 20)
EYE_SE  = sphere_to_cartesian(2.5, 60, -90 + 45)
EYE_ESE = sphere_to_cartesian(2.2, 70, -90 + 70)
EYE_E   = sphere_to_cartesian(2.2, 45, -90 + 90)



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


def new_map(*vfs, **kwargs):
    setdefaults(kwargs, 
                bgpath=RC['bgpath'],
                alpha=0.9,
                extent=RC['extent'])
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


def add_surface(*args, **kwargs):
    """FIXME"""
    setdefaults(kwargs,
                axes=(0, 1, 2),
                min_bounds=RC['min_bounds'],
                max_bounds=RC['max_bounds'],
                grid_shape=RC['grid_shape'],
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
                bgpath=RC['bgpath'],
                min_bounds=RC['min_bounds'],
                max_bounds=RC['max_bounds'],
                grid_shape=RC['grid_shape'])

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
