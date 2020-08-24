from bloptools.DE_opt_utils import run_fly_sim
from bloptools.DE_optimization import optimization_plan

# param_bounds = {'Aperture': {'horizontalSize': [1, 10],
#                              'verticalSize': [.1, 1]},
#                 'Lens': {'horizontalFocalLength': [10, 30]},
#                 'Obstacle': {'horizontalSize': [1, 10]}}

# param_bounds = {'Toroid': {'grazingAngle': [5, 10],
#                            'tangentialRadius': [1000, 10000]},
#                 'CM': {'grazingAngle': [5, 10]}}

# param_bounds = {'Toroid': {'grazingAngle': [5, 10]},
#                 'CM': {'grazingAngle': [5, 10]}}

param_bounds = {'Aperture': {'horizontalSize': [1, 10],
                             'verticalSize': [.1, 1]},
                'Lens': {'horizontalFocalLength': [10, 30]}}

# run with:
# RE(optimization_plan(fly_plan=run_fly_sim, bounds=param_bounds, db=db, run_parallel=True,
# num_interm_vals=1, num_scans_at_once=2, sim_id='Gu73sx1B', server_name='http://localhost:8000',
# root_dir=root_dir, watch_name='W60', flyer_name='sirepo_flyer', intensity_name='mean',
# opt_type='sirepo'))
