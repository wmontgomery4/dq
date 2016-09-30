import imp, sys, os, os.path, glob
import argparse

from base import *
from default_rc import rc

def plot_costs(costs):
    plt.plot(costs, 'b-', alpha=0.8, label='Training')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Total Cost')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.title('Episode Costs')

def plot_errors(errors):
    plt.semilogy(errors, 'b-', alpha=0.8, label='Training')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Mean Error')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.title('Bellman Errors')

def main():
    # Parse arguments and pass to main method
    parser = argparse.ArgumentParser(description='Run/analyze DeepQ experiments')
    parser.add_argument('name', type=str, help='Experiment name')
    parser.add_argument('--init', action='store_true', help='Initialize experiment')
    parser.add_argument('--train', action='store_true', help='Run new experiment')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing experiment')
    parser.add_argument('--plot-train', action='store_true', help='Plot training episodes from the experiment')
    parser.add_argument('--plot-test', action='store_true', help='Plot test episodes using learned agent')
    args = parser.parse_args()

    # Flags act on experiment directory "experiments/$name"
    expt_dir = 'dq_experiments/{}'.format(args.name)
    data_dir = '{}/data'.format(expt_dir)
    rc_file = '{}/rc.py'.format(expt_dir)

    # Exit 1) Initialize new directory
    if args.init:
        if os.path.exists(expt_dir):
            sys.exit("Experiment already exists")
        os.makedirs(expt_dir)
        os.makedirs(data_dir)
        with open('{}/rc.py'.format(expt_dir), 'w+') as f:
            f.write("# empty rc file")
        sys.exit()

    # Otherwise load the experiment runtime config
    override = imp.load_source('rc', rc_file)
    rc.update(override.rc)
    rc['expt_name'] = args.name
    rc['expt_dir'] = expt_dir
    rc['data_dir'] = data_dir

    # Run experiment (will also analyze)
    if args.train:
        # Double check before overwriting old experiment
        if os.path.exists(data_dir) and os.listdir(data_dir):
            yn = raw_input("Data already exists, would you like to overwrite (y/n)? ")
            if yn != 'y':
                sys.exit()
            for f in glob.glob('{}/*'.format(data_dir)):
                os.remove(f)
        expt = Experiment(rc)
        expt.run()

    # Exit 2) Analyze experiment
    if args.analyze or args.train:
        if rc['verbose']:
            print "Analyzing data"

        # Load up data files
        with open('{}/episodes.pkl'.format(data_dir), 'rb') as f:
            episodes = cPickle.load(f)

        # Extract things
        trajs = []
        costs = np.empty(0)
        errors = np.empty(0)
        for x0, U, cost, error in episodes:
            trajs.append( (x0, U) )
            costs = np.r_[costs, cost]
            errors = np.r_[errors, error]

        # Plot costs/errors for entire training run
        fig = plt.figure(figsize=(12,6))
        plt.suptitle(rc['title'], fontsize=18)
        plt.subplot(121)
        plot_costs(costs)
        plt.subplot(122)
        plot_errors(errors)
        plt.savefig('{}/costs_errors.png'.format(data_dir))
        plt.clf()

        # Plot nets/trajectories for each snapshot if method is provided
        if 'plot' in rc:
            # Load the world (for recreating trajectories)
            world = mj.MJCWorld(rc['model_file'])

            # run an analysis on each net, showing recent episodes and errors
            freq = rc['save_freq']

            for i in range(len(episodes) // freq):
                # get episode number and title
                n = (i+1)*freq
                title = "{}, {} Episodes".format(rc['title'], n)
                plt.suptitle(title, fontsize=18)

                # load network
                fname = '{}/nets_{}_episodes.pkl'.format(data_dir, n)
                with open(fname, 'rb') as f:
                    net, _ = cPickle.load(f)

                # plot and save
                rc['plot'](world, net, trajs[n-freq:n])
                plt.savefig('{}/{}_episodes.png'.format(data_dir, n))
                plt.clf()
        sys.exit()

    # Exit 3) Plot training episodes
    if args.plot_train:
        world = mj.MJCWorld(rc['model_file'])
        with open('{}/episodes.pkl'.format(data_dir), 'rb') as f:
            eps = cPickle.load(f)

        # Plot episodes until user quits
        idx = 0
        while True:
            # Handle user input
            input = raw_input("([q]uit/[n]ext) Which episode? ")
            if input == 'q':
                sys.exit()
            elif input == 'n':
                idx += 1
            else:
                try:
                    idx = int(input) - 1
                except TypeError:
                    print "Invalid option"
                    continue

            # Plot the episode
            x0, U = eps[idx][:2]
            world.plot(x0)
            for u in U:
                x1, _ = world.step(x0, u)
                x0 = x1
                world.plot(x0)

    # Exit 4) Plot testing episodes using most recent net
    # TODO: this won't work without adding pack_obs back to mjcpy2
    if args.plot_test:
        raise NotImplementedError("TODO: need to fix pack_obs for this")
#        world = mj.MJCWorld(rc['model_file'])
#        with open('{}/episodes.pkl'.format(data_dir), 'rb') as f:
#            eps = cPickle.load(f)
#        with open('{}/nets_{}_episodes.pkl'.format(data_dir, len(eps)), 'rb') as f:
#            net, _ = cPickle.load(f)
#
#        dX = eps[0][0].size
#        dO = 0
#        ndims = np.sum(rc['obs_dims'])
#        model = world.get_model()
#        fields = rc['obs_fields']
#        if 'qpos' in fields:
#            dO += model['nq']
#        if 'qvel' in fields:
#            dO += model['nv']
#        if 'xipos' in fields:
#            dO += ndims*(model['nbody'] - 1)
#        if 'ximat' in fields:
#            dO += ndims*ndims*(model['nbody'] - 1)
#        if 'site_xpos' in fields:
#            dO += ndims*model['nsite']
#        if 'to_target' in fields:
#            dO += ndims
#
#        # Run episodes until the user quits
#        while True:
#            x0 = rc['x0'] + rc['x0_noise']*2*(np.random.rand(dX) - 0.5)
#            world.plot(x0)
#
#            for i in range(rc['len_episode']):
#                obs = world.pack_obs(rc['obs_fields'], rc['obs_dims'], dO)
#                u = net.U(obs[None, :])[0]
#                x1, _ = world.step(x0, u)
#                x0 = x1
#                world.plot(x0)

    else:
        sys.exit("Run with '--help'")

if __name__ == '__main__':
    main()
