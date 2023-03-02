import os
USER = os.environ.get('USER')

import numpy as np
import tensorflow as tf 
import sys
sys.path.append('/projects/' + USER + '/igm/src')
from igm import Igm

# Linear SMB parameterization
class Igm(Igm):
    def update_smb_mysmb(self):
        ela = self.ELA 
        smb = (self.usurf - ela) * self.gradmb
        smb = tf.where(self.icemask > 0.5, smb, -10)
        self.smb.assign(smb)

glacier = Igm()

# Point to ice flow emulator 
glacier.config.iceflow_model_lib_path = '/projects/' + USER + '/igm/model-lib/f15_cfsflow_GJ_22_a/50'

# Point to geology file
glacier.config.geology_file = './inputs/geology.nc'

# Set up SMB parameters
glacier.config.type_mass_balance = 'mysmb'
glacier.ELA = 1200
glacier.gradmb = 0.003

# Ice thickness smoothing parameters
glacier.config.smoothing_thk_filter_shape = 3
glacier.config.smoothing_thk_sigma = 5
glacier.config.smoothing_thk_update_freq = 1

# Other configuration variables
glacier.config.tstart = 0
glacier.config.tend = 100
glacier.config.tsave = 1
glacier.config.cfl = 0.08
glacier.config.usegpu = True 
glacier.config.vars_to_save = ["topg", "usurf", "thk", "smb", "velbar_mag", "velsurf_mag", "divflux", "uvelsurf", "vvelsurf", "uvelbase", "vvelbase"]

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
        glacier.update_ncdf_ex()
        glacier.update_ncdf_ts()

glacier.print_all_comp_info()
