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

# Point to geology file
glacier.config.geology_file = './inputs/geology.nc'

# Set SMB parameterization
glacier.config.type_mass_balance = 'simple'

# Other configuration variables
glacier.config.tstart = 2020
glacier.config.tend = 2120
glacier.config.tsave = 5
glacier.config.usegpu = True 

# Initialize the model
glacier.initialize()

# Run the forward model
with tf.device("/GPU:0"):
    glacier.load_ncdf_data(glacier.config.geology_file)
    glacier.initialize_fields()

    while glacier.t < glacier.config.tend:
        glacier.update_smb()
        glacier.update_iceflow()
        glacier.update_t_dt()
        glacier.update_thk()

glacier.print_all_comp_info()
