from base import *

###################################
## default runtime configuration ##
###################################
rc = {
    'verbose' : True,
    'plot_episodes' : False,

    # Learning parameters
    'tau'       : 1e-3,
    'lr_actor'  : 1e-4,
    'lr_critic' : 1e-3,
    'l2_actor'  : 0.0,
    'l2_critic' : 0.0,
    'gradient_updates' : ls.updates.adam,

    # Net parameters
    'num_units'   : 64,
    'ctrl_limits' : None,

    # Observation parameters (all must be set by user)
    'obs_fields' : {},
    'obs_dims' : [True, True, True], # [X, Y, Z] dimensions observed
    'dO' : None,

    # Initialization parameters
    'x0'            : 0.0,
    'x0_noise'      : 1.0,

    # Exploration parameters
    'ctrl_noise'      : 1.0,
    'hold_ctrl_noise' : 1,

    # Cost parameters
    'gamma' : 0.99,
    'l2_q' : 0.0,
    'l2_v' : 0.0,
    'l2_u' : 0.0,
    'huber_alpha' : 1.0,
    'huber_site' : None,

    # Algorithm parameters
    'update_mode'   : 'online',
    'num_episodes'  : 1000,
    'len_episode'   : 100,
    'save_freq'     : 100,
    'num_batches'   : 100,
    'batch_size'    : 64,
}
