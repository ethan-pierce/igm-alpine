import os
USER = os.environ.get('USER')

import numpy as np
import tensorflow as tf 
import sys
sys.path.append('/projects/' + USER + '/igm/src')
from igm import Igm

glacier = Igm()

# Point to ice flow emulator 
glacier.config.iceflow_model_lib_path = '/projects/' + USER + '/igm/model-lib/f15_cfsflow_GJ_22_a/50'

# Point to observation file
glacier.config.observation_file = './inputs/observation.nc'

# Control and cost variables
glacier.config.opti_control = ['thk', 'strflowctrl', 'usurf']
glacier.config.opti_cost = ['velsurf', 'thk', 'usurf', 'icemask']

# Set tolerance
glacier.config.opti_velsurfobs_std = 5 # m/a
glacier.config.opti_thkobs_std = 5 # m
glacier.config.opti_usurfobs_std = 5 # m

# Regularization parameters
glacier.config.opti_regu_param_thk = 10
glacier.config.opti_regu_param_strflowctrl = 10

# Smoothing parameters
glacier.config.opti_smooth_anisotropy_factor = 0.2

# Save variables
glacier.config.opti_vars_to_save = ['topg', 'usurf', 'thk', 'strflowctrl', 'arrhenius', 'slidingco', 
                                    'velsurf_mag', 'velsurfobs_mag', 'divflux']

# Initialize inverse model
glacier.initialize()

# Run the optimization routine
with tf.device("/GPU:0"):
    glacier.load_ncdf_data(glacier.config.observation_file)
    glacier.initialize_fields()
    glacier.optimize()

glacier.print_all_comp_info()
