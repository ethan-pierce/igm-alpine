import numpy as np
import sys
import tensorflow as tf 
from igm import Igm

glacier = Igm()

# Point to ice flow emulator 
glacier.config.iceflow_model_lib_path = str(sys.argv[1])

# Point to observation file
glacier.config.observation_file = '../inputs/observation.nc'

# Control and cost variables
glacier.config.opti_control = ['thk', 'strflowctrl', 'usurf']
glacier.config.opti_cost = ['velsurf', 'thk', 'usurf', 'icemask']

# Set tolerance
glacier.config.opti_velsurfobs_std = 5 # m/a
glacier.config.opti_thkobs_std = 5 # m
glacier.config.opti_usurfobs_std = 5 # m

# Initialize inverse model
glacier.initialize()

# Run the optimization routine
with tf.device("/GPU:0"):
    glacier.load_ncdf_data(glacier.config.observation_file)
    glacier.initialize_fields()
    glacier.optimize()

glacier.print_all_comp_info()
