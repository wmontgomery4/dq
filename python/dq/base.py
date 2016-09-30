import cPickle
import mjcpy as mj
import numpy as np; np.random.seed(47)
import theano as th
import theano.tensor as tt
import lasagne as ls
import lasagne.layers as ll
from lasagne.nonlinearities import linear, tanh, ScaledTanh
import matplotlib.pyplot as plt

class ActorCriticNet():
    def __init__(self, dO, dU, num_units, ctrl_limits=None):
        self.dO = dO
        self.dU = dU
        self.O = tt.matrix()

        # actor
        obs_in = ll.InputLayer((None, dO), self.O)
        h1 = ll.DenseLayer(obs_in, num_units=num_units)
        h2 = ll.DenseLayer(h1, num_units=num_units)
        if ctrl_limits is not None:
            scaled_tanh = ScaledTanh(scale_out=ctrl_limits)
            self.actor = ll.DenseLayer(h2, num_units=dU, nonlinearity=scaled_tanh, W=ls.init.Uniform(1e-3))
        else:
            self.actor = ll.DenseLayer(h2, num_units=dU, nonlinearity=linear, W=ls.init.Uniform(1e-3))

        # critic (takes actor as input)
        h1 = ll.DenseLayer(obs_in, num_units=num_units)
        c1 = ll.ConcatLayer([h1, self.actor])
        h2 = ll.DenseLayer(c1, num_units=num_units)
        self.critic = ll.DenseLayer(h2, num_units=1, nonlinearity=linear, W=ls.init.Uniform(1e-3))

        # store all the params
        self.actor_params = ll.get_all_params(self.actor, trainable=True)
        self.all_params = ll.get_all_params(self.critic, trainable=True)
        self.critic_params = [p for p in self.all_params if p not in self.actor_params]

        # Helper functions for evaluating value/policy/Q
        # TODO: use 'deterministic=True' if/when adding batch_norm
        U = ll.get_output(self.actor)
        self.U = th.function([self.O], U)

        V = ll.get_output(self.critic)
        self.V = th.function([self.O], V)

        U = tt.matrix()
        Q = ll.get_output(self.critic, inputs={self.actor: U})
        self.Q = th.function([self.O, U], Q)

# TODO: make this more efficient
class ReplayBuffer():
    def __init__(self, dO, dU, init_size=100):
        # Initialize buffer
        self.dO = dO
        self.dU = dU
        self.obs0 = np.empty((init_size, dO))
        self.ctrl = np.empty((init_size, dU))
        self.cost = np.empty((init_size, 1))
        self.obs1 = np.empty((init_size, dO))

        # Explicitly store index of last stored memory
        self.T  = 0

    def add(self, obs0, ctrl, cost, obs1):
        # double buffer if there's no more room
        if self.T == self.obs0.shape[0]:
            self.obs0 = np.r_[self.obs0, np.empty((self.T, self.dO))]
            self.ctrl = np.r_[self.ctrl, np.empty((self.T, self.dU))]
            self.cost = np.r_[self.cost, np.empty((self.T, 1))]
            self.obs1 = np.r_[self.obs1, np.empty((self.T, self.dO))]

        self.obs0[self.T] = obs0
        self.ctrl[self.T] = ctrl
        self.cost[self.T] = cost
        self.obs1[self.T] = obs1
        self.T += 1

    def get_batch(self, size, idx=None):
        # usual choice, use idx if need deterministic batches for testing
        if idx is None:
            idx = np.random.choice(self.T, size)
        return (self.obs0[idx], self.ctrl[idx], self.cost[idx], self.obs1[idx])

class Experiment():
    def __init__(self, rc):
        # steal stuff from world/initialize
        self.rc = rc
        self.world = mj.MJCWorld(rc['model_file'])
        self.model = self.world.get_model()
        self.dX = self.model['nq'] + self.model['nv']
        self.dU = self.model['nu']

        # compute dO
        # TODO: put this somewhere else, reorganize observation module
        dO = 0
        ndims = np.sum(rc['obs_dims'])
        fields = rc['obs_fields']
        if 'qpos' in fields:
            dO += self.model['nq']
        if 'qvel' in fields:
            dO += self.model['nv']
        if 'xipos' in fields:
            dO += ndims*(self.model['nbody'] - 1)
        if 'ximat' in fields:
            dO += ndims*ndims*(self.model['nbody'] - 1)
        if 'site_xpos' in fields:
            dO += ndims*self.model['nsite']
        if 'to_target' in fields:
            dO += ndims
        self.dO = dO

        # TODO: start with synthetic data included/reorganize buffer storage
        self.buf = ReplayBuffer(self.dO, self.dU)

        # create nets and copy train net to target net
        self.net = ActorCriticNet(self.dO, self.dU, rc['num_units'], rc['ctrl_limits'])
        self.target_net = ActorCriticNet(self.dO, self.dU, rc['num_units'], rc['ctrl_limits'])
        values = ll.get_all_param_values(self.net.critic)
        ll.set_all_param_values(self.target_net.critic, values)

        # compile all the theano functions
        self._compile()

    def run(self):
        rc = self.rc
        self.episodes = []
        for i in range(rc['num_episodes']):
            # run episode and add experience to buffer/episodes
            x0, U, C, J = self._episode()
            self.episodes.append( (x0, U, C, J) )

            # update networks
            if rc['verbose']:
                print "{}, Ep {},\tCost: {:7.3f},\tAvg Bell Err: {:.6f}".format(rc['expt_name'], i+1, C, J.mean())

            # take snapshots intermittently
            if (i+1) % rc['save_freq'] == 0:
                if rc['verbose']:
                    print "Taking snapshot"
                self._save()

    def _compile(self):
        rc = self.rc

        # actor gradient step
        O = self.net.O
        V = ll.get_output(self.net.critic)
        params = self.net.actor_params
        regl_params = ll.get_all_params(self.net.actor, regularizable=True)
        regl = 0.5*rc['l2_actor']*tt.sum([tt.sum(p**2) for p in regl_params])
        updates = rc['gradient_updates'](V.mean()+regl, params, learning_rate=rc['lr_actor'])
        self.update_actor = th.function([O], [V.mean()], updates=updates)

        # critic bellman error (test version, doesn't update parameters)
        U = tt.matrix()
        Q = ll.get_output(self.net.critic, inputs={self.net.actor: U})
        Y = tt.matrix()
        J = 0.5*tt.mean((Y-Q)**2)
        self.J = th.function([O, U, Y], J)

        # critic bellman error (train version, does update parameters)
        regl_params = [p for p in ll.get_all_params(self.net.critic, regularizable=True)
                if p not in ll.get_all_params(self.net.actor)]
        regl = 0.5*rc['l2_critic']*tt.sum([tt.sum(p**2) for p in regl_params])
        params = self.net.critic_params
        updates = rc['gradient_updates'](J+regl, params, learning_rate=rc['lr_critic'])
        self.update_critic = th.function([O, U, Y], J, updates=updates)

        # target network update
        updates = []
        tau = rc['tau']
        for p,tgt_p in zip(self.net.all_params, self.target_net.all_params):
            updates.append( (tgt_p, tau*p + (1-tau)*tgt_p) )
        self.update_target = th.function([], [], updates=updates)

        # build cost function
        # TODO: handle this better through rc
        x = tt.vector()
        u = tt.vector()
        site_xpos = tt.matrix()

        # L2 costs
        c = 0.5*rc['l2_q']*tt.sum(x[:self.model['nq']]**2)
        c += 0.5*rc['l2_v']*tt.sum(x[-self.model['nv']:]**2)
        c += 0.5*rc['l2_u']*tt.sum(u**2)

        # Huber costs
        if rc['huber_site'] is not None:
            a = rc['huber_alpha']
            d = site_xpos[0] - site_xpos[1]
            c += rc['huber_site']*(tt.sqrt(tt.sum(d**2) + a**2) - a)

        # compile cost function
        # TODO: remove need for 'on_unused_input'
        self.cost = th.function([x, u, site_xpos], c, on_unused_input='ignore')

    def _pack_obs(self):
        """ Pack the current world state into observation vector. """
        # TODO: add this back into mjcpy2, way faster.
        rc = self.rc
        fields = rc['obs_fields']
        dims = rc['obs_dims']

        data = self.world.get_data()
        model = self.world.get_model()

        obs = np.array([])
        if 'qpos' in fields:
            obs = np.r_[obs, data['qpos'][0]]
        if 'qvel' in fields:
            obs = np.r_[obs, data['qvel'][0]]
        if 'xipos' in fields:
            xipos = data['xipos'][1:] # Skip worldbody.
            for pos in xipos:
                for j in range(3): # Might mask some of dims
                    if dims[j]:
                        obs = np.r_[obs, pos[j]]
        if 'ximat' in fields:
            ximat = data['ximat'][1:] # Skip worldbody.
            for mat in ximat:
                for j in range(3): # Might mask some of dims
                    for k in range(3):
                        if dims[j] and dims[k]:
                            obs = np.r_[obs, mat[3*j+k]]
        if 'site_xpos' in fields:
            sites = data['site_xpos']
            for site in sites:
                for j in range(3): # Might mask some of dims
                    if dims[j]:
                        obs = np.r_[obs, site[j]]
        if 'to_target' in fields:
            sites = data['site_xpos']
            assert sites.shape[0] == 2
            diff = sites[1] - site[0]
            for j in range(3): # Might mask some of dims
                if dims[j]:
                    obs = np.r_[obs, diff[j]]

        assert obs.size == self.dO
        return obs

    def _episode(self):
        rc = self.rc

        # Init new buffers
        T = rc['len_episode']
        O = np.empty((T+1, self.dO))
        X = np.empty((T+1, self.dX))
        U = np.empty((T, self.dU))
        C = np.empty(T)
        if rc['update_mode'] == 'online':
            J = np.empty(T)

        # Initialize episode
        x0 = rc['x0'] + rc['x0_noise']*2*(np.random.rand(self.dX) - 0.5)
        q0 = x0[:self.model['nq']]
        v0 = x0[-self.model['nv']:]
        self.world.set_data({"qpos" : q0, "qvel": v0})
        self.world.kinematics()

        # Run an episode, storing costs/errors
        X[0] = x0
        O[0] = self._pack_obs()
        for t in range(T):
            # take step with current net and store transition
            # note: O[t, t+1] makes sure that input is 2D (since net takes batches)
            U[t] = self.net.U(O[t:t+1])
            if t % rc['hold_ctrl_noise'] == 0:
                noise = rc['ctrl_noise']*np.random.randn(self.dU)
            U[t] += noise

            if rc['ctrl_limits'] is not None:
                U[t] = np.minimum(U[t], rc['ctrl_limits'])
                U[t] = np.maximum(U[t], -rc['ctrl_limits'])

            X[t+1], site_xpos = self.world.step(X[t], U[t])
            O[t+1] = self._pack_obs()

            # TODO: change rc['cost'] to make this better
            C[t] = self.cost(X[t+1], U[t], site_xpos)
            self.buf.add(O[t], U[t], C[t], O[t+1])

            # update now if doing online updates
            if rc['update_mode'] == 'online':
                J[t] = self._update()

        # update afterwards otherwise
        if rc['update_mode'] != 'online':
            J = np.array([self._update() for _ in range(rc['num_batches'])])

        if rc['plot_episodes']:
            for x in X:
                self.world.plot(x)

        # store everything necessary to recreate the episode, plus cost and error
        return (X[0], U, C.sum(), J)

    def _update(self):
        # perform all 3 network updates
        obs0, ctrl, cost, obs1 = self.buf.get_batch(self.rc['batch_size'])
        Y = cost + self.rc['gamma']*self.target_net.V(obs1)
        J = self.update_critic(obs0, ctrl, Y)
        self.update_actor(obs0)
        self.update_target()
        return J

    def _save(self):
        # pickle nets and episode
        nets = (self.net, self.target_net)
        data_dir = self.rc['data_dir']
        with open('{}/nets_{}_episodes.pkl'.format(data_dir, len(self.episodes)), 'w+b') as f:
            cPickle.dump(nets, f, -1)
        with open('{}/episodes.pkl'.format(data_dir), 'w+b') as f:
            cPickle.dump(self.episodes, f, -1)
