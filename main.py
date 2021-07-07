from networks.SNN_V2.SNN_v2 import evaluate
from environments.grid_world_frames.grid_env_frames_gym import GridEnvFrames
from binarynet.train_binary_net_from_frames import train_binary_net
import matplotlib.pyplot as plt

if __name__ == '__main__':

    hyperparams = dict(N_hidden=1000,
                       N_place=100,
                       input_hidden_connectivity_params={'in_h_w_on_prob': 0.05,
                                                         'in_h_w_off_prob': 0.1,
                                                         'on_amplitude': 1.0,
                                                         'off_amplitude': 0.25},
                       h_place_w_tau_s=2e3,
                       place_act_w_tau_s=1e3,
                       place_v_noise=0.4,
                       act_v_noise=0.4,
                       act_v_noise_min=0.1,
                       act_v_noise_dec_steps=1e4,
                       tau_e_action=1,
                       tau_e_action_max=10,
                       tau_e_action_inc_steps=1e5,
                       tau_e_place=1,
                       tau_e_place_max=40,
                       tau_e_place_inc_steps=1e5
                       )

    units_per_dim = 10

    train_binaryNet = True

    image_size = 64
    dense_layer_size = 128

    binary_model_name = 'Linnaeus_5_' + str(image_size) + 'x' + str(image_size) + '_' + str(
        dense_layer_size) + '_model'

    train_img_path = 'binarynet/Img_DB/Linnaeus 5 256x256/other/*.jpg'

    if train_binaryNet:
        train_binary_net(img_path=train_img_path, model_name=binary_model_name, resolution=image_size,
                         dense_layer_size=dense_layer_size, debug=False)

    env = GridEnvFrames(units_per_dim=units_per_dim, dimensions=2, action_per_dim=3, maze_config=4, img_folders=[
        'binarynet/Img_DB/Linnaeus 5 256x256/berry/*.jpg',
        'binarynet/Img_DB/Linnaeus 5 256x256/bird/*.jpg',
        'binarynet/Img_DB/Linnaeus 5 256x256/dog/*.jpg',
        'binarynet/Img_DB/Linnaeus 5 256x256/flower/*.jpg'],
                        model_name=binary_model_name,
                        image_size=image_size, cont_action=True, position_as_input=False, debug_level=0)

    print('Environment observation space:', env.observation_space)
    print('Environment action space:', env.action_space)
    print()

    trace_latency = evaluate(hyperparams, env, action_dims=2, action_per_dim=3, total_episodes=1000)

    plt.show()