"""Run an inverse method, using IGM, over Mendenhall Glacier."""

import os
USER = os.environ.get('USER')

import sys
sys.path.append('/projects/' + USER + '/igm/src')

import numpy as np
import tensorflow as tf
from igm import Igm

####################################
# Section 1: Run the inverse model #
####################################

glacier = Igm()

# Point to ice flow emulator 
glacier.config.iceflow_model_lib_path = '/projects/' + USER + '/igm/model-lib/f15_cfsflow_GJ_22_a/50'

# Point to observation file
glacier.config.observation_file = './inputs/observation.nc'

# Set control and cost variables
glacier.config.opti_control = ['thk', 'strflowctrl', 'usurf']
glacier.config.opti_cost = ['velsurf', 'thk', 'usurf', 'icemask']

# Set weights on control and cost variables
glacier.config.opti_strflowctrl_std = 20.0
glacier.config.opti_thr_strflowctrl = 100

glacier.config.opti_usurfobs_std = 5.0
glacier.config.opti_thkobs_std = 20.0

# Set initial sliding regime
glacier.config.init_strflowctrl = 100

# Set regularization parameters
glacier.config.opti_regu_param_thk = 10.0
glacier.config.opti_regu_param_strflowctrl = 5.0

# Set smoothing and convexity parameters
glacier.config.opti_smooth_anisotropy_factor = 0.2
glacier.config.opti_convexity_weight = 0.002

# Set computational parameters
glacier.config.opti_nbitmax = 2000
glacier.config.opti_output_freq = 50
glacier.config.opti_step_size = 0.01
glacier.config.opti_init_zero_thk = False

# Choose variables to save
glacier.config.opti_vars_to_save = ['topg', 'usurf', 'thk', 'strflowctrl', 'arrhenius', 'slidingco', 
                                    'velsurf_mag', 'velsurfobs_mag', 'divflux', 
                                    'uvelbase', 'vvelbase', 'uvelsurf', 'vvelsurf']

# Initialize inverse model
glacier.initialize()

# Run the optimization routine
with tf.device("/GPU:0"):
    glacier.load_ncdf_data(glacier.config.observation_file)
    glacier.initialize_fields()
    glacier.optimize()

Ub = glacier.getmag(glacier.uvelbase, glacier.vvelbase)
Us = glacier.getmag(glacier.uvelsurf, glacier.vvelsurf)

output_dir = './outputs/scalar_thkobs_std/'
np.savetxt(output_dir + 'sliding_velocity_magnitude.txt', Ub)
np.savetxt(output_dir + 'surface_velocity_magnitude.txt', Us)
