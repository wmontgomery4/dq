from base import *

rc = {
    'title'        : "2D Particle, Huber Loss, Ctrl \in [-10,10], \\tau=0.1",
    'model_file'   : "mjc_models/particle2d.xml",

    # world params (cost/noise is default)
    'ctrl_limits'  : 10.0,

    # observation params
    'obs_fields'   : {
        'qpos'     : True,
        'qvel'     : True,
    },

    # cost params
    'l2_u' : 1e-2,
    'huber_site' : 1.0,
    'huber_alpha' : 0.1,

    # algorithm params (see defaults for these as well)
    'tau' : 0.1,
    'num_episodes' : 200,
    'save_freq'    : 25,
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
    batch = np.zeros((grid_size**2, 4))
    batch[:, 0] = XX.flatten()
    batch[:, 1] = YY.flatten()
    U = net.U(batch)
    V = net.V(batch).reshape((grid_size, grid_size))

    # plot policy
    plt.subplot(131, aspect='equal')
    plt.quiver(batch[:, 0], batch[:, 1], U[:, 0], U[:, 1])
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title("Policy (0 Velocity)")

    # plot value
    plt.subplot(132)
    plt.imshow(V, origin='lower', extent=[-bound,bound,-bound,bound])
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title("Value (0 Velocity)")
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
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title("Recent Sample Trajectories")

# add to rc
rc['plot'] = plot
