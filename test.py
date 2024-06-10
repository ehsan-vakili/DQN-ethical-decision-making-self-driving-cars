import os
import datetime
import time
import random
import numpy as np
import tensorflow as tf
import csv
from src.dqn_agent import DQNAgent
from src.carla_environment import CarlaEnv
from config import get_args
from src.uncertainty import *

Number_of_test_episodes = 10

weather_params_base = {
    'cloudiness': 0,
    'sun_altitude_angle': 90
}


def evaluate_dqn_agents(config):
    """
    Function to evaluate DQN agents on different scenarios and uncertainty checks.

    Args:
    - config (dict): Configuration parameters for the DQN agents.

    Returns:
    - None
    """

    model_base_dir = "models"
    knob_values = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]

    env = CarlaEnv(**config)
    state_shape = env.observation_space.shape
    action_dim = env.action_space.n

    env.change_lighting_conditions(weather_params_base)

    if config['uncertainty_check']:
        num_samples = config['sample_nets']  # number of sample networks (T)
        env.change_lighting_conditions(weather_params)

    for knob_value in knob_values:
        model_dir = os.path.join(model_base_dir, str(knob_value))
        model_path = os.path.join(model_dir, 'final_model.h5')
        buffer_path = os.path.join(model_dir, 'final_replay_buffer.pkl')

        if not os.path.exists(model_path) or not os.path.exists(buffer_path):
            print(f"Model or buffer not found for knob value: {knob_value}")
            continue
        else:
            print(f"-----------------Testing trained agent with k={knob_value}-----------------")

        agent_config = {
            'input_shape': state_shape,
            'num_actions': action_dim,
            'replay_buffer_capacity': config['replay_buffer_capacity'],
            'batch_size': config['batch_size'],
            'gamma': config['gamma'],
            'lr': config['lr'],
            'epsilon_start': config['epsilon_start'],
            'epsilon_end': config['epsilon_end'],
            'epsilon_decay': config['epsilon_decay'],
            'target_update': config['target_update'],
            'save_model_freq': config['save_model_freq'],
            'knob_value': knob_value,
            'log_dir': ""
        }
        agent = DQNAgent(**agent_config)
        agent.load_model(model_path)
        agent.load_replay_buffer(buffer_path)

        if config['uncertainty_check']:
            ground_truth_model = build_cnn_model_with_dropout(state_shape, action_dim, config['lr'], 0)
            ground_truth_model.set_weights(agent.policy_net.get_weights())

            # Optimize phi
            phi_list = []
            noisy_image_list = []
            state = env.reset(evaluate=2)
            done = False
            finished = False
            print("\n----------Finding the best phi value; it may take a while!----------\n")
            while not finished:
                for i in range(num_samples):
                    noisy_image = env.get_noisy_image()
                    noisy_image_list.append(noisy_image)
                y_gt = ground_truth_model.predict(np.expand_dims(state, axis=0))[0]
                best_phi, best_nll, best_rmse = optimize_phi(state_shape, action_dim, config['lr'], num_samples, y_gt, noisy_image_list)
                phi_list.append(best_phi)

                action = agent.act_trained(state)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                if done:
                    finished = True
                    break
            env.destroy_actors()
            phi = max(set(phi_list), key=phi_list.count)
            print(f"\nBest dropout rate (phi): {phi}")

            # Build Bayesian model with best phi
            bayesian_model = build_cnn_model_with_dropout(state_shape, action_dim, config['lr'], phi)
            bayesian_model.set_weights(agent.policy_net.get_weights())

        total_rewards_1 = []
        total_rewards_2 = []

        inference_times = []

        with open(os.path.join(model_dir, f'{knob_value}_1.csv'), 'w', newline='') as csvfile1:
            fieldnames = ['Episode', 'Ego Location_x', 'Ego Location_y', 'Velocity_x', 'Velocity_y', 'Acceleration_x',
                          'Acceleration_y', 'Action', 'Time']
            writer1 = csv.DictWriter(csvfile1, fieldnames=fieldnames)
            writer1.writeheader()

            print("\n--------------Testing in scenario 1----------------\n")

            for e in range(Number_of_test_episodes):
                state = env.reset(evaluate=1)
                done = False
                total_reward = 0
                finished = False
                state_time = datetime.datetime.now()

                while not finished:
                    if config['uncertainty_check']:
                        for i in range(num_samples):
                            noisy_image = env.get_noisy_image()
                            noisy_image_list.append(noisy_image)
                        y_gt = ground_truth_model.predict(np.expand_dims(state, axis=0))[0]
                        sigma_tot, mu_bar, y_pred = calculate_total_uncertainty(bayesian_model, noisy_image_list, num_samples)
                        position = np.argmax(y_gt)
                        y_gt = y_gt[position]
                        mu_bar = mu_bar[position]
                        y_predict = [y[position] for y in y_pred]
                        sigma_tot = sigma_tot[position]

                        nll = calculate_nll(sigma_tot, y_gt, mu_bar)
                        rmse = calculate_rmse(y_gt, y_predict)

                        print(f"NLL: {nll}, RMSE: {rmse}")

                    start_time = time.time()
                    action = agent.act_trained(state)
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    total_reward += reward

                    ego_location = env.get_location()
                    velocity = env.get_velocity()
                    acceleration = env.get_acceleration()
                    t = datetime.datetime.now() - state_time
                    writer1.writerow({'Episode': e+1, 'Ego Location_x': ego_location[0], 'Ego Location_y': ego_location[1],
                                      'Velocity_x': velocity[0], 'Velocity_y': velocity[1],
                                      'Acceleration_x': acceleration[0], 'Acceleration_y': acceleration[1],
                                      'Action': action, 'Time': t.total_seconds()})
                    if done:
                        env.destroy_actors()
                        print(f"Scenario 1 - Test episode {e+1} - Collision with: {env.get_collision_history()}")
                        finished = True
                        break
                total_rewards_1.append(total_reward)
                print(f"Scenario 1 - Total reward in test episode {e+1}: {total_reward}")
            print(f"Scenario 1 - Knob value: {knob_value}, Averaged total reward: {np.mean(total_rewards_1)}\n")

        with open(os.path.join(model_dir, f'{knob_value}_2.csv'), 'w', newline='') as csvfile2:
            fieldnames = ['Episode', 'Ego Location_x', 'Ego Location_y', 'Velocity_x', 'Velocity_y', 'Acceleration_x',
                          'Acceleration_y', 'Action', 'Time']
            writer2 = csv.DictWriter(csvfile2, fieldnames=fieldnames)
            writer2.writeheader()

            print("\n--------------Testing in scenario 2----------------\n")
            for e in range(Number_of_test_episodes):
                state = env.reset(evaluate=2)
                done = False
                total_reward = 0
                finished = False
                state_time = datetime.datetime.now()

                while not finished:
                    if config['uncertainty_check']:
                        for i in range(num_samples):
                            noisy_image = env.get_noisy_image()
                            noisy_image_list.append(noisy_image)
                        y_gt = ground_truth_model.predict(np.expand_dims(state, axis=0))[0]
                        sigma_tot, mu_bar, y_pred = calculate_total_uncertainty(bayesian_model, noisy_image_list, num_samples)
                        position = np.argmax(y_gt)
                        y_gt = y_gt[position]
                        mu_bar = mu_bar[position]
                        y_predict = [y[position] for y in y_pred]
                        sigma_tot = sigma_tot[position]

                        nll = calculate_nll(sigma_tot, y_gt, mu_bar)
                        rmse = calculate_rmse(y_gt, y_predict)

                        print(f"NLL: {nll}, RMSE: {rmse}")

                    start_time = time.time()
                    action = agent.act_trained(state)
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    total_reward += reward

                    ego_location = env.get_location()
                    velocity = env.get_velocity()
                    acceleration = env.get_acceleration()
                    t = datetime.datetime.now() - state_time
                    writer2.writerow({'Episode': e+1, 'Ego Location_x': ego_location[0], 'Velocity_x': velocity[0],
                                      'Ego Location_y': ego_location[1], 'Velocity_y': velocity[1],
                                      'Acceleration_x': acceleration[0], 'Acceleration_y': acceleration[1],
                                      'Action': action, 'Time': t.total_seconds()})
                    if done:
                        env.destroy_actors()
                        print(f"Scenario 2 - Test episode {e+1} - Collision with: {env.get_collision_history()}")
                        finished = True
                        break
                total_rewards_2.append(total_reward)
                print(f"Scenario 2 - Total reward in test episode {e+1}: {total_reward}")
            print(f"Scenario 2 - Knob value: {knob_value}, Averaged total reward: {np.mean(total_rewards_2)}\n")

        # Calculate the mean inference time for each knob-value trained model
        mean_inference_time = np.mean(inference_times)
        print(f"\nMean inference time per step for knob value={knob_value} over {Number_of_test_episodes} episodes:"
              f"{mean_inference_time*1000} ms\n")


if __name__ == "__main__":
    config = get_args()
    evaluate_dqn_agents(config)
