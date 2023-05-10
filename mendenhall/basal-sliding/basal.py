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
glacier.config.opti_strflowctrl_std = 5.0
glacier.config.opti_velsurfobs_std = 2.5
glacier.config.opti_thkobs_std = 25.0
glacier.config.opti_usurfobs_std = 5.0

# Set initial sliding regime
glacier.config.init_slidingco = 0
glacier.config.init_arrhenius = 78

# Set regularization parameters
glacier.config.opti_regu_param_thk = 10.0
glacier.config.opti_regu_param_strflowctrl = 5.0

# Set smoothing and convexity parameters
glacier.config.opti_smooth_anisotropy_factor = 0.2
glacier.config.opti_convexity_weight = 0.002

# Set computational parameters
glacier.config.opti_nbitmax = 1500
glacier.config.opti_output_freq = 50
glacier.config.opti_step_size = 0.001
glacier.config.opti_init_zero_thk = False

# Choose variables to save
glacier.config.opti_vars_to_save = ['topg', 'usurf', 'thk', 'strflowctrl', 'arrhenius', 'slidingco', 
                                    'velsurf_mag', 'velsurfobs_mag', 'divflux']

# Initialize inverse model
glacier.initialize()

# Run the optimization routine
with tf.device("/GPU:0"):
    glacier.load_ncdf_data(glacier.config.observation_file)
    glacier.initialize_fields()
    glacier.optimize()

# glacier.print_all_comp_info()

####################################
# Section 2: Run the forward model #
####################################

# Configure a new instance of IGM
model = Igm()

# Point to ice flow emulator 
model.config.iceflow_model_lib_path = '/projects/' + USER + '/igm/model-lib/f15_cfsflow_GJ_22_a/50'

# Point to geology file
model.config.geology_file = './geology-optimized.nc'

# Configure the model
model.config.usegpu = True
model.config.vars_to_save = ["topg", "usurf", "thk", "velbar_mag", "velsurf_mag", "divflux", "uvelsurf", "vvelsurf", "uvelbase", "vvelbase"]
model.config.tstart = 0.0
model.config.tend = 0.1
model.config.tsave = 0.1

# Initialize the model
model.initialize()

# Run the forward model
with tf.device("/GPU:0"):
    model.load_ncdf_data(model.config.geology_file)
    model.initialize_fields()
    model.update_iceflow()
    model.update_ncdf_ex()
    model.update_ncdf_ts()

# model.print_all_comp_info()
