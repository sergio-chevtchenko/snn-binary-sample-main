import numpy as np
import matplotlib.pyplot as plt

class SNN:
    def __init__(self, env, N_input, params, action_dims=None, action_per_dim=None):
        self.env = env

        if action_per_dim is not None:
            self.action_per_dim = action_per_dim
            self.N_action = action_per_dim * action_dims
            self.action_dims = action_dims
        else:
            self.action_per_dim = None
            self.N_action = self.env.action_space.n

        self.N_input = N_input
        self.N_hidden = params['N_hidden']
        self.N_place = params['N_place']

        # hidden neurons
        self.hidden_v = np.zeros(self.N_hidden).astype(np.float32)  # voltage
        self.hidden_th = 1.0  # threshold
        self.hidden_trace = np.zeros(self.N_hidden).astype(np.float32)  # spike trace

        # input -> hidden synapses
        self.in_h_w = self.generate_in_h_w(self.N_input, self.N_hidden, params['input_hidden_connectivity_params'])

        # place neurons
        self.place_v = np.zeros(self.N_place).astype(np.float32)  # voltage
        self.place_trace = np.zeros(self.N_place).astype(np.float32)  # spike trace

        # hidden -> place synapses
        self.h_place_w = np.zeros((self.N_hidden, self.N_place)).astype(np.float32)
        self.h_place_z = np.zeros((self.N_hidden, self.N_place)).astype(np.float32)  # eligibility trace

        # action neurons
        self.act_v = np.zeros(self.N_action).astype(np.float32)

        # place -> action synapses
        self.place_act_w = np.zeros((self.N_place, self.N_action)).astype(np.float32)
        self.place_act_z = np.zeros((self.N_place, self.N_action)).astype(np.float32)  # eligibility trace

        self.tau_e_place = params['tau_e_place']
        self.tau_e_action = params['tau_e_action']

        self.h_place_w_tau_s = params['h_place_w_tau_s']

        self.place_v_noise = params['place_v_noise']
        self.act_v_noise = params['act_v_noise']
        self.act_v_noise_min = params['act_v_noise_min']
        self.act_v_noise_dec_rate = (self.act_v_noise - self.act_v_noise_min) / params['act_v_noise_dec_steps']

        self.place_act_w_tau_s = params['place_act_w_tau_s']

        self.tau_e_action_max = params['tau_e_action_max']
        self.tau_e_place_max = params['tau_e_place_max']

        self.tau_e_action_inc_rate = (self.tau_e_action_max - self.tau_e_action) / params['tau_e_action_inc_steps']
        self.tau_e_place_inc_rate = (self.tau_e_place_max - self.tau_e_place) / params['tau_e_place_inc_steps']


    def generate_in_h_w(self, N_input, N_hidden, connectivity_params):
        """
            returns input-hidden weights, as encoded by the connectivity parameters

            :param N_input: input layer size
            :param N_hidden: hidden layer size
            :param connectivity_params: defines sparsity and amplitude of connectivity
            """

        in_h_w = np.zeros((N_input, N_hidden)).astype(np.float32)
        for i in range(N_input):
            n_on = int(connectivity_params['in_h_w_on_prob'] * N_hidden)
            n_off = int(connectivity_params['in_h_w_off_prob'] * N_hidden)

            start_l = int(((i / N_input) * N_hidden) - connectivity_params['on_amplitude'] * N_hidden)
            stop_l = int(((i / N_input) * N_hidden) + connectivity_params['on_amplitude'] * N_hidden)

            if stop_l >= N_hidden:
                d = stop_l - N_hidden + 1
                start_l -= d
                stop_l -= d
            elif start_l < 0:
                d = -start_l
                start_l += d
                stop_l += d

            a = np.arange(start=max(0, start_l), stop=min(N_hidden - 1, stop_l))

            np.random.shuffle(a)
            in_h_w[i, a[0:n_on]] = 1.0

            start_l = int(((i / N_input) * N_hidden) - connectivity_params['off_amplitude'] * N_hidden)
            stop_l = int(((i / N_input) * N_hidden) + connectivity_params['off_amplitude'] * N_hidden)

            if stop_l >= N_hidden:
                d = stop_l - N_hidden + 1
                start_l -= d
                stop_l -= d
            elif start_l < 0:
                d = -start_l
                start_l += d
                stop_l += d

            a = np.arange(start=max(0, start_l), stop=min(N_hidden - 1, stop_l))

            np.random.shuffle(a)
            in_h_w[i, a[0:n_off]] = -1

        return in_h_w

    def step_net(self, in_signal, rw_hp=0, rw_pa=0):
        """
            Single step pass through the network

            :param in_signal: A binary array of size N_input
            :param rw_hp: Hidden-place layer reward signal
            :param rw_pa: Place-action layer reward signal
            """

        assert len(in_signal) == self.N_input

        # Input spikes, this is a list of neuron indices
        spiked_input = np.where(in_signal == 1)[0]

        # propagate input -> hidden spike
        self.hidden_v += np.sum(self.in_h_w[spiked_input], axis=0)

        spiked_hidden = np.where(self.hidden_v >= self.hidden_th)[0]

        # reset all neurons after the spike
        self.hidden_v *= 0

        self.hidden_trace *= 0
        self.hidden_trace += -1
        self.hidden_trace[spiked_hidden] = 1

        # hidden -> place weight update
        self.h_place_w += self.h_place_z * rw_hp
        self.h_place_w = np.clip(self.h_place_w, -10, 1.0)

        self.h_place_w *= 1 - 1 / self.h_place_w_tau_s

        # Place neurons
        # propagate hidden -> place spike
        self.place_v += np.sum(self.h_place_w[spiked_hidden], axis=0)
        # add noise
        self.place_v += np.random.normal(0, self.place_v_noise, (self.N_place)).astype(np.float32)

        # spikes only the neuron with the highest voltage
        spiked_place = [np.argmax(self.place_v)]


        # print('place_v:\n', self.place_v)

        # resets the spiked place neuron potential to -1
        self.place_v[spiked_place] = -1

        # clips the voltage of place neurons
        self.place_v = np.clip(self.place_v, -1.0, 1.0)

        self.place_trace *= 0
        self.place_trace[spiked_place] = 1

        #self.skip_count = self.place_act_steps[spiked_place][0]

        # print('spiked hidden/place: ', spiked_hidden, spiked_place)

        # hidden -> place trace

        # update the eligibility trace
        self.h_place_z *= 1 - 1 / self.tau_e_place
        #self.h_place_z[:, spiked_place] = self.h_place_z[:, spiked_place] + self.hidden_trace[:, None]
        self.h_place_z[:, spiked_place] += self.hidden_trace[:, None]

        np.clip(a=self.h_place_z, a_min=-1.0, a_max=1.0, out=self.h_place_z)

        # place -> action weigh update
        self.place_act_w += self.place_act_z * rw_pa
        self.place_act_w = np.clip(self.place_act_w, 0, 1)

        self.place_act_w *= 1 - 1 / self.place_act_w_tau_s

        # propagate place->action spike
        self.act_v += np.sum(self.place_act_w[spiked_place], axis=0)

        # action neurons

        self.act_v += np.random.normal(0, self.act_v_noise, (self.N_action))

        # print('act_v:', self.act_v)
        if self.action_per_dim is not None:
            sp = []
            for i in range(self.action_dims):
                sp.append(
                    int(i * self.action_per_dim) + np.argmax(
                        self.act_v[int(i * self.action_per_dim): int((i + 1) * self.action_per_dim)]))

            # if np.random.rand() < self.epsilon:
            #     sp = []
            #     for i in range(self.N_dim):
            #         sp.append(int(i * self.action_per_dim) + np.random.randint(self.action_per_dim))
        else:
            sp = [np.argmax(self.act_v)]

            # if np.random.rand() < self.epsilon:
            #     sp = [np.random.randint(self.N_action)]

        # print('act_v:', self.act_v)
        self.act_v *= 0.0

        spiked_action = sp.copy()

        # place -> action trace
        self.place_act_z *= 1 - 1 / self.tau_e_action

        # post (action) spike
        #self.place_act_z[:, spiked_action] = self.place_act_z[:, spiked_action] + self.place_trace[:, None]
        self.place_act_z[:, spiked_action] += self.place_trace[:, None]

        self.place_act_z[self.place_act_z > 1] = 1

        # print('place_act_z:\n', self.place_act_z)

        return spiked_action, spiked_hidden, spiked_place

    def learn(self, total_episodes, verbose=False, debug=False):

        trace_episode_latency = []

        for episode in range(total_episodes):
            # reset the environment before each episode
            state = self.env.reset()

            reward_hp = 0
            reward_pa = 0

            flag_break = False

            latency = 0
            while True:
                self.tau_e_action += self.tau_e_action_inc_rate
                if self.tau_e_action > self.tau_e_action_max:
                    self.tau_e_action = self.tau_e_action_max

                self.tau_e_place += self.tau_e_place_inc_rate
                if self.tau_e_place > self.tau_e_place_max:
                    self.tau_e_place = self.tau_e_place_max

                self.act_v_noise -= self.act_v_noise_dec_rate
                if self.act_v_noise < self.act_v_noise_min:
                    self.act_v_noise = self.act_v_noise_min

                latency += 1
                # get action from the SNN
                spiked_action, _, _ = self.step_net(in_signal=state,
                                              rw_hp=reward_hp,
                                              rw_pa=reward_pa)

                # print('spiked_action:', spiked_action)

                if flag_break:
                    break

                if self.action_per_dim is not None:
                    action = spiked_action.copy()
                    for d in range(self.action_dims):
                        action[d] = action[d] - d * self.action_per_dim - 1
                else:
                    action = spiked_action[0]

                state, rew, done, info = self.env.step(action, render=False)

                if rew == 1:
                    reward_hp = 1
                    reward_pa = 1

                    flag_break = True
                elif done:
                    reward_hp = 0
                    reward_pa = 0
                    flag_break = True
                else:
                    reward_hp = 0
                    reward_pa = 0

            trace_episode_latency.append(latency)

            plt.clf()
            plt.subplot(211)
            plt.plot(trace_episode_latency)
            plt.xlabel('Episode #')
            plt.ylabel('Latency')

            plt.subplot(212)

            for [x, y] in self.env.obstacles:
                self.env.grid[x, y] = 3
                cax = plt.imshow(self.env.grid, cmap='jet')

            plt.pause(0.001)

def evaluate(hyperparams, env, action_dims, action_per_dim, total_episodes):
    N_input = env.observation_space.shape[0]

    model = SNN(env, N_input, hyperparams, action_dims=action_dims, action_per_dim=action_per_dim)

    print('training...')
    # Train the agent
    model.learn(total_episodes=total_episodes, verbose=False, debug=False)

    return env.trace_latency

