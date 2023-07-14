"""Run an inverse method, using IGM, over Mendenhall Glacier."""

import os
USER = os.environ.get('USER')

import sys
sys.path.append('/projects/' + USER + '/igm/src')

import numpy as np
import tensorflow as tf
from igm import Igm

# Set up data assimilation module
glacier = Igm()
glacier.config.version = 'v2'
glacier.config.iceflow_model_lib_path = '/projects/' + USER + '/igm/model-lib/f21_pinnbp_GJ_23_a'
glacier.config.observation_file = '../inputs/observation.nc'

# Set control and cost variables
glacier.config.opti_control = ['thk', 'slidingco', 'usurf']
glacier.config.opti_cost = ['velsurf', 'thk', 'usurf', 'icemask']

# Save most variables
glacier.config.opti_vars_to_save = [
    'topg', 'usurf', 'thk', 
    'strflowctrl', 'arrhenius', 'slidingco', 
    'velsurf_mag', 'velsurfobs_mag', 'divflux', 
    'uvelbase', 'vvelbase', 'uvelsurf', 'vvelsurf'
]

# Set optimization parameters
glacier.config.opti_nbitmax = 2000
glacier.config.opti_output_freq = 100
glacier.config.opti_step_size_v2 = 0.01
glacier.config.opti_smooth_anisotropy_factor = 0.2

# Set weight for the regularization terms
glacier.config.opti_regu_param_thk = 5.0
glacier.config.opti_regu_param_slidingco = 10.0

# Set confidence levels on observations
glacier.config.opti_usurfobs_std = 10.0
glacier.config.opti_velsurfobs_std = 20.0
glacier.config.opti_thkobs_std = 20.0

# Set parameters for sliding behavior
glacier.opti_slidingco_std = 5000.0
glacier.config.opti_thr_slidingco = 78000
glacier.config.init_slidingco = 78000

# Initialize and run the optimization scheme
glacier.initialize()

with tf.device("/GPU:0"):
    glacier.load_ncdf_data(glacier.config.observation_file)
    glacier.initialize_fields()
    glacier.update_iceflow_emulated()
    glacier.optimize()

Ub = glacier.getmag(glacier.uvelbase, glacier.vvelbase)
Us = glacier.getmag(glacier.uvelsurf, glacier.vvelsurf)

output_dir = './'
np.savetxt(output_dir + 'sliding_velocity_magnitude.txt', Ub)
np.savetxt(output_dir + 'surface_velocity_magnitude.txt', Us)
