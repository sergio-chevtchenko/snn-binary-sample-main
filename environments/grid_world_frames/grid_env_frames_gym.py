import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from keras.models import Model
import glob
import cv2
import progressbar
import datetime

class GridEnvFrames(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, units_per_dim, dimensions, action_per_dim, maze_config, img_folders=None, model_name=None, image_size=256, cont_action=False,
                 position_as_input=False, debug_level=0):
        super(GridEnvFrames, self).__init__()
        self.units_per_dim = units_per_dim
        self.action_per_dim = action_per_dim
        self.maze_config = maze_config
        self.target = np.ones((2)) * int(self.units_per_dim / 2 - 1)
        self.obstacles = []
        self.model_name = model_name
        self.debug_level = debug_level

        self.image_size = image_size
        self.position_as_input = position_as_input

        if self.maze_config == 1:
            # conf_1
            self.target = np.ones((2)) * int(self.units_per_dim / 2 - 1)
            self.target[1] = self.units_per_dim - 1
        elif self.maze_config == 2:
            # conf_2
            self.target = np.ones((2)) * int(self.units_per_dim / 2 - 1)
            self.target[1] = self.units_per_dim - 1
        elif self.maze_config == 3:
            # conf_3
            for y in range(int(self.units_per_dim / 4), int(self.units_per_dim - self.units_per_dim / 4)):
                x = int(self.units_per_dim / 4)
                self.obstacles.append([x, y])
            for y in range(int(self.units_per_dim / 4), int(self.units_per_dim - self.units_per_dim / 4)):
                x = int(self.units_per_dim - self.units_per_dim / 4)
                self.obstacles.append([x, y])
            for x in range(int(self.units_per_dim / 4), 1 + int(self.units_per_dim - self.units_per_dim / 4)):
                y = int(self.units_per_dim / 4)
                self.obstacles.append([x, y])
        elif self.maze_config == 4:
            # conf_4
            for y in range(int(self.units_per_dim / 4), int(self.units_per_dim - self.units_per_dim / 4)):
                x = int(self.units_per_dim / 4)
                self.obstacles.append([x, y])
            for y in range(int(self.units_per_dim / 4), int(self.units_per_dim - self.units_per_dim / 4)):
                x = int(self.units_per_dim - self.units_per_dim / 4)
                self.obstacles.append([x, y])
            for x in range(int(self.units_per_dim / 4), 1 + int(self.units_per_dim - self.units_per_dim / 4)):
                y = int(self.units_per_dim / 4)
                self.obstacles.append([x, y])
        elif self.maze_config == 5:
            # debug
            self.target = np.ones((2)) * int(self.units_per_dim - 1)


        self.current_pos = np.zeros(2).astype(np.int)
        self.previous_pos = self.current_pos.copy()

        self.place_grid = np.zeros((self.units_per_dim, self.units_per_dim))
        self.freq_grid = np.zeros((self.units_per_dim, self.units_per_dim))

        self.latency = 0

        self.grid = np.zeros((units_per_dim, units_per_dim))

        self.trace_latency = []

        self.cont_action = cont_action
        # Define action and observation space
        # They must be gym.spaces objects
        if self.cont_action:
            self.action_space = spaces.Box(low=np.ones(int(2)) * -1,
                                           high=np.ones(int(2)), dtype=np.int)
        else:
            self.action_space = spaces.Discrete(int(action_per_dim ** 2))

        if self.position_as_input is False:
            if self.model_name is not None:
                # using a binary CNN to extract features
                model = load_model(self.model_name)

                print()
                print('Leaning from binary model ' + self.model_name)
                print()

                layer_name = 'act5'
                self.intermediate_layer_model = Model(inputs=model.input,
                                                      outputs=model.get_layer(layer_name).output)

                self.obs_state_size = self.intermediate_layer_model.output_shape[1]

                self.observation_space = spaces.Box(low=np.zeros(self.obs_state_size),
                                                    high=np.ones(self.obs_state_size),
                                                    dtype=np.uint8)
            else:
                # returning RGB images directly
                self.observation_space = spaces.Box(low=np.zeros((self.image_size, self.image_size, 3)),
                                                    high=np.ones((self.image_size, self.image_size, 3))*255,
                                                    dtype=np.uint8)

            # load random images from the dataset
            self.img_bank = []
            self.binary_out_bank = []
            # file_list = glob.glob("Linnaeus_5_64X64_test/*.jpg")
            file_list = []
            for img_folder in img_folders:
                files = glob.glob(img_folder)
                for file in files:
                    file_list.append(file)

            np.random.shuffle(file_list)

            print()
            print('Loading observations from the image bank...')
            print()

            bar = progressbar.ProgressBar(max_value=units_per_dim * units_per_dim)

            for i, frame in enumerate(file_list):
                bar.update(i)
                if i == units_per_dim * units_per_dim:
                    break

                img = cv2.imread(frame)
                img = cv2.resize(img, (image_size, image_size))

                if self.model_name is not None:
                    test = np.random.randn(1, self.image_size, self.image_size, 3)
                    test[0, :, :, :] = img / 255

                    self.img_bank.append(test.copy())

                    self.binary_out_bank.append(self.get_binary_CNN_observation(i))
                else:
                    self.img_bank.append(img.copy())
        else:
            self.observation_space = spaces.Box(low=np.zeros(2 * self.units_per_dim),
                                                high=np.ones(2 * self.units_per_dim),
                                                dtype=np.uint8)

        self.reset()

    def get_discrete_state(self, state):

        bin_str = np.zeros(self.observation_space.shape).astype(np.int)

        for i, s in enumerate(state):
            if state[i] < self.obs_min[i]:
                state[i] = self.obs_min[i]
            elif state[i] > self.obs_max[i]:
                state[i] = self.obs_max[i]

        discrete_state = ((state - self.obs_min) / self.discrete_os_win_size).astype(np.int)

        for i, d in enumerate(discrete_state):
            try:
                bin_str[int(d + i * self.units_per_dim_input)] = 1
            except:
                print()
                print('Exception! state:', state)
                print(discrete_state)
                print(i, d)
                print()

        self.discrete_state = discrete_state.copy()

        return bin_str, discrete_state

    def get_binary_CNN_observation(self, img_id):
        x = self.img_bank[img_id]

        intermediate_output = self.intermediate_layer_model.predict(x)[0]
        intermediate_output[intermediate_output == -1] = 0
        intermediate_output = intermediate_output.astype(np.uint)

        if self.debug_level > 1:
            cv2.imshow('frame', x[0, :, :, :])
            cv2.waitKey(1)

        return intermediate_output.astype(np.uint)

    def get_elapsed_episodes(self):
        return len(self.trace_latency)

    def reset(self):
        if len(self.trace_latency) % 100 == 0:
            print(datetime.datetime.now(), 'Episode #', len(self.trace_latency))
        if self.latency > 0:
            self.trace_latency.append(self.latency)
            
            if self.debug_level == 1:
                plt.clf()

                plt.subplot(211)

                for [x, y] in self.obstacles:
                    self.grid[x, y] = 3
                cax = plt.imshow(self.grid, cmap='jet')

                plt.subplot(212)

                plt.plot(self.trace_latency)

                plt.pause(0.001)
                

        self.latency = 0

        self.grid = np.zeros((self.units_per_dim, self.units_per_dim))

        if self.maze_config == 1:
            # conf_1
            self.current_pos = np.zeros(2).astype(np.int)
        elif self.maze_config == 2:
            # conf_2
            self.current_pos = np.zeros(2).astype(np.int)
            self.current_pos[0] = np.random.randint(self.units_per_dim - 1)
        elif self.maze_config == 3:
            # conf_3
            self.current_pos = np.zeros(2).astype(np.int)
        elif self.maze_config == 4:
            # conf_4
            self.current_pos = np.zeros(2).astype(np.int)
            #self.current_pos[0] = np.random.randint(self.units_per_dim - 1)
            self.current_pos[0] = np.random.randint(2) * (self.units_per_dim - 1)
            self.current_pos[1] = np.random.randint(2) * (self.units_per_dim - 1)
        elif self.maze_config == 5:
            # debug
            self.current_pos = np.zeros(2).astype(np.int)

        self.previous_pos = self.current_pos.copy()

        return self._next_observation()

    def _next_observation(self):
        if self.position_as_input is False:
            img_id = self.current_pos[0] + self.current_pos[1] * self.units_per_dim

            if self.model_name is not None:
                # state from a binary CNN
                #current_state = self.get_binary_CNN_observation(img_id)
                current_state = self.binary_out_bank[img_id]
            else:
                # state directly from the image bank
                current_state = self.img_bank[img_id]

                if self.debug_level > 1:
                    cv2.imshow('frame', current_state)
                    cv2.waitKey(1)

            return current_state
        else:
            # state as position
            current_state = np.zeros(int(self.units_per_dim * 2)).astype(np.uint16)
            for d in range(2):
                current_state[self.current_pos[d] + d * self.units_per_dim] = 1

            return current_state

    def numberToBase(self, n, b):
        if n == 0:
            return [0]
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return digits[::-1]

    def step(self, action, render=False):
        self.latency += 1

        # Execute one time step within the environment
        self._take_action(action)

        done = False
        reward = 0

        if np.linalg.norm(self.current_pos - self.target) < 0.5:
            done = True
            reward = 1
        elif self.latency >= (int(np.power(self.units_per_dim, 2) / 1)):
            done = True

        obs = self._next_observation()

        self.grid *= 0.95
        self.grid[self.previous_pos[0], self.previous_pos[1]] = 0.5
        self.grid[self.current_pos[0], self.current_pos[1]] = 1

        self.grid[int(self.target[0]), int(self.target[1])] = 2

        if render:
            self.render()

        if self.debug_level>1:
            self.render()

        return obs, reward, done, {}

    def _take_action(self, action):
        self.previous_pos = self.current_pos.copy()

        if self.cont_action:
            assert self.action_per_dim == 3

            for d in range(2):
                if action[d] < -0.33:
                    self.current_pos[d] -= 1
                elif action[d] > 0.33:
                    self.current_pos[d] += 1

        else:
            conv_act = self.numberToBase(action, self.action_per_dim)
            while len(conv_act) < 2:
                conv_act.insert(0, 0)

            conv_act = np.array(conv_act) - 1

            if self.action_per_dim == 3:
                self.current_pos += conv_act
            elif self.action_per_dim == 2:
                if action % 2 == 0:
                    self.current_pos[int(action / 2)] -= 1
                else:
                    self.current_pos[int(action / 2)] += 1

        for d in range(2):
            if self.current_pos[d] < 0:
                self.current_pos[d] = 0
            elif self.current_pos[d] >= self.units_per_dim:
                self.current_pos[d] = self.previous_pos[d]

        for [x, y] in self.obstacles:
            if self.current_pos[0] == x and self.current_pos[1] == y:
                self.current_pos = self.previous_pos.copy()
                break

    def render(self, mode='human', close=False, grid_aux=None):
        plt.clf()

        if grid_aux is None:
            plt.subplot(211)

            for [x, y] in self.obstacles:
                self.grid[x, y] = 3
            cax = plt.imshow(self.grid, cmap='jet')

            plt.subplot(212)

            plt.plot(self.trace_latency)
        else:
            plt.subplot(221)

            for [x, y] in self.obstacles:
                self.grid[x, y] = 3
            cax = plt.imshow(self.grid, cmap='jet')

            plt.subplot(222)

            cax = plt.imshow(grid_aux, cmap='jet')

            plt.subplot(223)

            plt.plot(self.trace_latency)

        plt.pause(0.001)
        #plt.show()

