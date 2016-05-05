from base import *

rc = {
    'title'         : "Pendulum Swing-Up, \\tau=1.0",
    'model_file'    : 'mjc_models/pendulum.xml',

    # world params
    'x0_noise'      : np.array([np.pi, 0.0]),
    'ctrl_limits'   : 1.0,

    'ctrl_noise'      : 0.5,
    'hold_ctrl_noise' : 25,

    # cost
    'l2_u' : 0.1,
    'l2_v' : 1.0,
    'huber_site' : 10.0,
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
    'tau' : 1.0,
    'num_episodes' : 300,
    'len_episode'  : 1500,
    'save_freq'    : 25,
}

## define function for analyzing the data
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
        wrapped = np.mod(X[0], 2*np.pi)
        plt.plot(wrapped, X[1], 'k--')
    plt.xlim([0, 2*np.pi])
    plt.ylim([-3, 3])
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title("Recent Sample Trajectories")

# add to rc
rc['plot'] = plot
