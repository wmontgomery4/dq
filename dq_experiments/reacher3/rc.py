from base import *

rc = {
    'title'         : "Three-Link Reacher",
    'model_file'    : 'mjc_models/reacher3.xml',
    'plot_episodes' : False,

    # world params
    'x0_noise'      : np.array([2.0, 0.5, 0.5, 0.3, 0.3, 0, 0, 0, 0, 0]),
    'ctrl_limits'   : 10,
    'ctrl_noise'    : 1.0,
    'hold_ctrl_noise' : 10,

    # observation params
    'obs_dims'      : [True, True, False], # X/Y
    'obs_fields'    : {
        'qvel'      : True,
        'xipos'     : True,
        'ximat'     : True,
        'site_xpos' : True,
        'to_target' : True,
    },

    # cost params
    'l2_u' : 1e-2,
    'huber_site' : 1.0,
    'huber_alpha' : 1e-2,

    # algorithm params
    'num_episodes'  : 1000,
    'len_episode'   : 500,
    'save_freq'     : 25,
}
