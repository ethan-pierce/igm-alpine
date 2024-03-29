import os
USER = os.environ.get('USER')

import sys
sys.path.append('/projects/' + USER + '/igm2')

# Import the most important libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import igm
 
# Select one OPTION btw the first, keep the MANDATORY ones, un/comment OPTIONAL modules
modules = [
        #    "prepare_data",       # OPTION 1  : download and prepare the data with OGGM
           "load_ncdf_data",      # OPTION 2  : read 2d data from netcdf files
        #    "load_tif_data",      # OPTION 3  : read 2d data from tif files
        #    "make_synthetic",     # OPTION 4  : make a synthetic glacier with ideal geom.
           "optimize",            # OPTIONAL  : optimize unobservable variables from obs.
        #    "mysmb",               # OPTIONAL  : custom surface mass balance model
           "flow_dt_thk",         # MANDATORY : does update iceflow, time step and thickness
           "vertical_iceflow",    # OPTIONAL  : computes vertical velocity
        #    "write_ncdf_ex",       # OPTIONAL  : write 2d state data to netcdf files
        #    "write_plot2d",       # OPTIONAL  : write 2d state plots to png files
           "print_info",          # OPTIONAL  : print basic live-info about the model state
           "print_all_comp_info", # OPTIONAL  : report information about computation time
        #    "anim3d_from_ncdf_ex", # OPTIONAL  : make a nice 3D animation of glacier evolution
          ]

# Collect and parse all the parameters of all model components
parser = igm.params_core()
for module in modules:
    getattr(igm, "params_" + module)(parser)
params = parser.parse_args()

# Override parameters
# params.RGI = 'RGI60-11.01450' # necessary when using prepare_data module
params.tstart = 2000.0
params.tend   = 2000.0
params.opti_nbitmax = 500
params.tsave  = 50
params.plot_live = False
params.observation = True

################################################# ETHAN'S PLAYGROUND :-) #################################################

params.opti_control = ["thk", "slidingco"]   # here you may add "usurf" and "slidingco"
params.opti_cost    = ["velsurf", "thk", "icemask"] # here you may add  "usurf" "thk" "divfluxfcz"
params.opti_regu_param_thk        = 10       # control the strength of the regularization for the bedrock
params.opti_step_size             = 1        # control the step size of the optimization
params.opti_regu_param_slidingco  = 10       # weight for the regul. of slidingco (if selected in controls)
params.opti_convexity_weight      = 0        # weight for the convexity term (zero is fine here)
params.opti_velobs_std            = 5.0     # standard deviation of the velocity observations
params.opti_usurfobs_std          = 5.0      # standard deviation of the surface elevation (if selected in controls)
params.opti_thkobs_std            = 25.0     # standard deviation of the ice thickness (if selected in controls)
params.init_slidingco             = 10000    # initial value for the sliding coefficient

########################################################################################################################

# Define a state class/dictionnary that contains all the data
state = igm.State()

# Place the computation on your device GPU ('/GPU:0') or CPU ('/CPU:0')
with tf.device("/GPU:0"):

    # Initialize all the model components in turn
    for module in modules:
        getattr(igm, "init_" + module)(params, state)

    # Time loop, perform the simulation until reaching the defined end time
    while state.t < params.tend:
        
        # Update each model components in turn
        for module in modules:
            getattr(igm, "update_" + module)(params, state)
            
    # Finalize each module in turn
    for module in modules:
        getattr(igm, "final_" + module)(params, state)
