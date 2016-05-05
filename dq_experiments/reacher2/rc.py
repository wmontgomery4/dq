from base import *

rc = {
    'title'         : "Two-Link Reacher",
    'model_file'    : 'mjc_models/reacher2.xml',

    # world params
    'x0_noise'      : np.array([np.pi, 3, 0.2, 0.2, 0, 0, 0, 0]),
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
    'tau' : 0.1,
    'num_episodes'  : 1000,
    'len_episode'   : 500,
    'save_freq'     : 25,
}

## define function for analyzing the data
num_plot_episodes = 6
def plot(world, net, trajs):
    fig = plt.gcf()
    fig.set_size_inches(6, 6)

    # plot trajectories
    for t in range(num_plot_episodes):
        x0, U = trajs[-t-1]
        world.set_data({"qpos": x0[:4], "qvel": x0[-4:]})
        diff = world.pack_obs({"to_target": True}, [True, True, False], 2)
        plt.scatter(diff[0], diff[1], marker='^', c='k')
        for u in U:
            x1, _ = world.step(x0, u)
            x0 = x1
            diff = np.c_[diff, world.pack_obs({"to_target": True}, [True, True, False], 2)]
        plt.plot(diff[0], diff[1], 'k--')
    plt.xlim([-0.3, 0.3])
    plt.ylim([-0.3, 0.3])
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title("Recent Sample Trajectories (relative to target)")

# add to rc
rc['plot'] = plot
