from math import pi

RC = {}

## 4D or 5D model ##
RC['model'] = 'Bicycle4D'

if RC['model'] == 'Bicycle5D':
    RC['min_bounds'] = [-1.2, -1.2, -pi, -pi/5, +0.3]
    RC['max_bounds'] = [+1.2, +1.2, +pi, +pi/5, +0.8]
    RC['grid_shape'] = (31, 31, 25, 7, 7)
elif RC['model'] == 'Bicycle4D':
    RC['min_bounds'] = [-1.2, -1.2, -pi, +0.3]
    RC['max_bounds'] = [+1.2, +1.2, +pi, +0.8]
    RC['grid_shape'] = (31, 31, 25, 7)

RC['entry_locations']    = ['entry_s', 'entry_w', 'entry_n', 'entry_e']
RC['exit_locations']     = ['exit_s', 'exit_w', 'exit_n', 'exit_e']
RC['locations']          = [
    'center', 
    'road_s', 'road_w', 'road_n', 'road_e',
    *RC['entry_locations'], 
    *RC['exit_locations']
]
RC['permitted_routes']   = {
    # South entrance
    ('entry_s', 'exit_w'): ('road_n', 'road_w'),
    ('entry_s', 'exit_n'): ('road_n', 'road_n'),
    ('entry_s', 'exit_e'): ('road_n', 'road_e'),
    # West entrance
    ('entry_w', 'exit_n'): ('road_e', 'road_n'),
    ('entry_w', 'exit_e'): ('road_e', 'road_e'),
    ('entry_w', 'exit_s'): ('road_e', 'road_s'),
    # North entrance
    ('entry_n', 'exit_w'): ('road_s', 'road_w'),
    ('entry_n', 'exit_s'): ('road_s', 'road_s'),
    ('entry_n', 'exit_e'): ('road_s', 'road_e'),
    # East entrance
    ('entry_e', 'exit_s'): ('road_w', 'road_s'),
    ('entry_e', 'exit_w'): ('road_w', 'road_w'),
    ('entry_e', 'exit_n'): ('road_w', 'road_n'),
    # U-turns
    ('entry_s', 'exit_s'): ('road_n', 'road_s'),
    ('entry_w', 'exit_w'): ('road_e', 'road_w'),
    ('entry_n', 'exit_n'): ('road_s', 'road_n'),
    ('entry_e', 'exit_e'): ('road_w', 'road_e'),
}

RC['extent'] = [RC['min_bounds'][0], RC['max_bounds'][0], 
                RC['min_bounds'][1], RC['max_bounds'][1]]

RC['bgpath'] = '../data/4way.png'

RC['bob'] = {}
RC['bob']['interactive']        = False
RC['bob']['data_dir']           = '../data'
RC['bob']['time_step']          = 0.2
RC['bob']['time_horizon']       = 5.
RC['bob']['max_window']         = 2.
RC['bob']['model']              = RC['model']
RC['bob']['grid_shape']         = RC['grid_shape']
RC['bob']['min_bounds']         = RC['min_bounds']
RC['bob']['max_bounds']         = RC['max_bounds']