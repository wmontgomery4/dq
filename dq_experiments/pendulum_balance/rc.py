from base import *

rc = {
    'title'         : "Pendulum Balance",
    'model_file'    : 'mjc_models/pendulum.xml',

    # world params
    'x0_noise'        : np.array([0.5, 0.1]),
    'ctrl_limits'     : 3.0,
    'ctrl_noise'      : 0.5,
    'hold_ctrl_noise' : 1,

    # cost
    'l2_u' : 1e-2,
    'huber_site' : 1.0,
    'huber_alpha' : 0.1,

    # observation params
    'obs_dims'      : [True, False, True], # X/Z
    'obs_fields'    : {
        'qvel'      : True,
        'xipos'     : True,
        'ximat'     : True,
        'site_xpos' : True,
    },

    # algorithm params
    'tau': 0.1,
    'num_episodes' : 100,
    'len_episode'  : 500,
    'save_freq'    : 25,
}

## define function for analyzing the data
bound = np.pi
grid_size = 25
num_plot_episodes = rc['save_freq']
def plot(world, net, trajs):
    fig = plt.gcf()
    fig.set_size_inches(6, 6)

    '''
    # TODO: write easy way to plot these...
    # NOTE: (YY, XX) instead of (XX, YY) because we're going to be using imshow
    YY, XX = np.mgrid[-bound:bound:grid_size*1j, -bound:bound:grid_size*1j]
    batch = np.c_[XX.flatten(), YY.flatten()]
    U = net.U(batch).reshape((grid_size, grid_size))
    V = net.V(batch).reshape((grid_size, grid_size))

    # plot policy
    plt.subplot(131)
    plt.imshow(U, origin='lower', extent=[-bound,bound,-bound,bound])
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title("Policy")
    plt.colorbar()

    # plot value
    plt.subplot(132)
    plt.imshow(V, origin='lower', extent=[-bound,bound,-bound,bound])
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title("Value")
    plt.colorbar()
    '''

    # plot trajectories
    for i in range(num_plot_episodes):
        x0, U = trajs[-i]
        X = x0
        for u in U:
            x1, _ = world.step(x0, u)
            X = np.c_[X, x0]
            x0 = x1
        wrapped = np.mod(X[0] + np.pi, 2*np.pi) - np.pi
        plt.plot(wrapped, X[1], 'k--')
    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.1, 0.1])
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title("Recent Sample Trajectories")

# add to rc
rc['plot'] = plot
