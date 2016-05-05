from base import *

rc = {
    'title'        : "1D Particle, Quadratic Loss, Control \in [-3, 3], \\tau = 1.0",
    'model_file'   : "mjc_models/particle1d.xml",

    # world params (cost/noise is default)
    'ctrl_limits'  : 3.0,

    # observation params
    'obs_fields'   : {
        'qpos'     : True,
        'qvel'     : True,
    },

    # cost params
    'l2_q' : 1.0,
    'l2_u' : 1e-2,

    # algorithm params (see defaults for these as well)
    'tau' : 1.0,
    'num_episodes'     : 100,
    'save_freq'        : 25,
    'gradient_updates' : ls.updates.nesterov_momentum
}

## define function for analyzing the data
bound = 4
grid_size = 25
num_plot_episodes = rc['save_freq']
def plot(world, net, trajs):
    fig = plt.gcf()
    fig.set_size_inches(13, 6)

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

    # plot trajectories
    plt.subplot(133, aspect='equal')
    for i in range(num_plot_episodes):
        x0, U = trajs[-i]
        X = x0
        for u in U:
            x1, _ = world.step(x0, u)
            X = np.c_[X, x0]
            x0 = x1
        plt.plot(X[0], X[1], 'k--')
    plt.xlim([-bound, bound])
    plt.ylim([-bound, bound])
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title("Recent Sample Trajectories")

# add to rc
rc['plot'] = plot
