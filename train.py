import time
import os
import datetime
import numpy as np
import tensorflow as tf
from src.dqn_agent import DQNAgent
from src.carla_environment import CarlaEnv
from config import get_args


def train_dqn_agent(config):
    """
    Function to train a DQN agent in the CARLA environment.

    Args:
    - config (dict): Configuration parameters for training.

    Returns:
    - None
    """

    # Extract knob value from configuration
    knob_value = config['knob_value']
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + f"-knob-{knob_value}")
    model_dir = os.path.join("models", f"{knob_value}")

    # Create directories for logging and model saving
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Initialize CARLA environment and extract state and action dimensions
    env = CarlaEnv(**config)
    state_shape = env.observation_space.shape
    action_dim = env.action_space.n

    # Configure DQN agent
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
        'log_dir': log_dir
    }
    agent = DQNAgent(**agent_config)

    # Training loop
    for e in range(config['episodes']):  # Total training episodes
        state = env.reset(evaluate=0)
        done = False
        total_reward = 0
        finished = False

        while not finished:  # Training steps
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            clipped_reward = min(1, max(-1, reward))
            agent.store_experience(state, action, clipped_reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                env.destroy_actors()
                finished = True
                break

            agent.train()

        agent.train_episode += 1
        agent.rewards.insert(0, total_reward)
        if len(agent.rewards) > 100:
            agent.rewards = agent.rewards[:-1]
        avg_rewards = np.mean(agent.rewards)
        print(f"Episode: {e + 1}/{config['episodes']}, Total reward: {total_reward}, Average reward: {avg_rewards}")

        # Logging average reward
        with agent.summary_writer.as_default():
            tf.summary.scalar('average_reward', avg_rewards, step=e)

        # Evaluation phase
        print(f"train step: {agent.train_step}")
        if (agent.train_step // config['evaluation_interval']) != agent.evaluation_checkpoint:
            print("\n******Evaluation*********\n")
            evaluate_agent(agent, env, config['evaluation_steps'])
            print("\n******Training*********\n")

    # Save final model and replay buffer
    final_model_path = os.path.join(model_dir, 'final_model.h5')
    final_replay_buffer_path = os.path.join(model_dir, 'final_replay_buffer.pkl')
    agent.save_model(final_model_path)
    agent.save_replay_buffer(final_replay_buffer_path)


def evaluate_agent(agent, env, evaluation_steps):
    """
    Function to evaluate a trained agent in the CARLA environment.

    Args:
    - agent (DQNAgent): The trained DQN agent.
    - env (CarlaEnv): The CARLA environment.
    - evaluation_steps (int): Number of evaluation steps.

    Returns:
    - None
    """
    # Evaluation in the first scenario
    state = env.reset(evaluate=1)
    total_reward = 0

    for _ in range(evaluation_steps):
        action = agent.act_trained(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        if done:
            print(f"Evaluation episode {agent.evaluation_episode_1} (First Scenario),"
                  f"total reward: {total_reward}, collision: {env.get_collision_history()}")
            env.destroy_actors()
            time.sleep(0.5)
            with agent.summary_writer.as_default():
                tf.summary.scalar('collision_history_eval1', env.get_collision_history(), step=agent.evaluation_episode_1)
                tf.summary.scalar('Episode_total_reward_eval1', total_reward, step=agent.evaluation_episode_1)
            agent.evaluation_episode_1 += 1
            total_reward = 0
            state = env.reset(evaluate=1)

    env.destroy_actors()

    # Evaluation in the second scenario
    total_reward = 0
    state = env.reset(evaluate=2)
    for _ in range(evaluation_steps):
        action = agent.act_trained(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        if done:
            print(f"Evaluation episode {agent.evaluation_episode_2} (Second Scenario),"
                  f"total reward: {total_reward}, collision: {env.get_collision_history()}")
            env.destroy_actors()
            time.sleep(0.5)
            with agent.summary_writer.as_default():
                tf.summary.scalar('collision_history_eval2', env.get_collision_history(), step=agent.evaluation_episode_2)
                tf.summary.scalar('Episode_total_reward_eval2', total_reward, step=agent.evaluation_episode_2)
            agent.evaluation_episode_2 += 1
            total_reward = 0
            state = env.reset(evaluate=2)

    agent.evaluation_checkpoint += 1


if __name__ == "__main__":
    config = get_args()  # Get configuration parameters
    train_dqn_agent(config)  # Train the DQN agent
