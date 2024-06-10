import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from collections import deque
import os
import datetime
import pickle

tf.keras.utils.disable_interactive_logging()


class ReplayBuffer:
    """
    Replay buffer to store and sample experience tuples for training.
    """
    def __init__(self, capacity):
        """
        Initialize the replay buffer.

        Parameters:
        capacity (int): Maximum number of experiences to store in the buffer.
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def insert(self, state, action, reward, next_state, done):
        """
        Insert a new experience into the buffer.

        Parameters:
        state (np.array): The current state.
        action (int): The action taken.
        reward (float): The reward received.
        next_state (np.array): The next state.
        done (bool): Whether the episode is done.
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.buffer = pickle.load(f)


def build_cnn_model(input_shape, num_actions, learning_rate):
    """
    Build a CNN model for Q-value approximation.

    Parameters:
    input_shape (tuple): Shape of the input state.
    num_actions (int): Number of possible actions.
    learning_rate (float): Learning rate for the optimizer.

    Returns:
    tf.keras.Model: The compiled CNN model.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Lambda(lambda layer: layer / 255)(inputs)
    x = layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), activation='relu',
                      kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), activation='relu',
                      kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
    predictions = layers.Dense(num_actions, activation='linear',
                               kernel_initializer=initializers.VarianceScaling(scale=2.))(x)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model


class DQNAgent:
    """
    Deep Q-Network agent for training and interacting with the environment.
    """
    def __init__(self, **kwargs):
        """
        Initialize the DQN agent with the given parameters.

        Parameters:
        kwargs (dict): Dictionary of configuration parameters.
        """
        self.input_shape = kwargs['input_shape']
        self.num_actions = kwargs['num_actions']
        self.gamma = kwargs['gamma']
        self.batch_size = kwargs['batch_size']
        self.epsilon_start = kwargs['epsilon_start']
        self.epsilon_end = kwargs['epsilon_end']
        self.epsilon_decay = kwargs['epsilon_decay']
        self.target_update = kwargs['target_update']
        self.save_checkpoint = kwargs['save_model_freq']
        self.log_dir = kwargs['log_dir']
        self.knob = kwargs['knob_value']

        self.epsilon = self.epsilon_start
        self.replay_buffer = ReplayBuffer(kwargs['replay_buffer_capacity'])
        self.policy_net = build_cnn_model(self.input_shape, self.num_actions, kwargs['lr'])
        self.target_net = build_cnn_model(self.input_shape, self.num_actions, kwargs['lr'])
        self.update_target_network()

        self.train_step = 0
        self.evaluation_checkpoint = 0

        self.train_episode = 0
        self.evaluation_episode_1 = 0
        self.evaluation_episode_2 = 0

        self.rewards = []

        if bool(self.log_dir):
            self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
            self.summary_writer = tf.summary.create_file_writer(self.log_dir)
            # self.summary_writer.set_as_default()

    def update_target_network(self):
        """
        Update the target network with the weights from the policy network.
        """
        self.target_net.set_weights(self.policy_net.get_weights())

    def act(self, state):
        """
        Choose an action based on the current state.
        Considers epsilon for exploration-exploitation trade-off.

        Parameters:
        state (np.array): The current state.

        Returns:
        int: The action to take.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        state = np.expand_dims(state, axis=0)
        act_values = self.policy_net.predict(state)
        return np.argmax(act_values[0])

    def act_trained(self, state):  # No exploration
        """
        Choose an action without exploration (for evaluation).

        Parameters:
        state (np.array): The current state.

        Returns:
        int: The action to take.
        """
        state = np.expand_dims(state, axis=0)
        act_values = self.policy_net.predict(state)
        return np.argmax(act_values[0])

    def train(self):
        """
        Train the policy network with experiences from the replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        target_q = self.policy_net.predict(states)
        next_q = self.target_net.predict(next_states)

        for i in range(self.batch_size):
            target_q[i][actions[i]] = rewards[i] if dones[i] else rewards[i] + self.gamma * np.amax(next_q[i])

        self.policy_net.fit(states, target_q, epochs=1, verbose=0, callbacks=[self.tensorboard_callback])
        self.train_step += 1

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        if self.train_step % self.target_update == 0:
            self.update_target_network()

        if self.train_episode % self.save_checkpoint == 0:
            self.save_model(f'{self.log_dir}/model_checkpoint_{self.train_episode}.h5')
            self.save_replay_buffer(f'{self.log_dir}/replay_buffer_{self.train_episode}.pkl')

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.insert(state, action, reward, next_state, done)

    def save_model(self, filename):
        print("Model saving...")
        self.policy_net.save(filename)
        print("Model saved.")

    def load_model(self, filename):
        self.policy_net = tf.keras.models.load_model(filename)
        self.update_target_network()

    def save_replay_buffer(self, filename):
        print("Buffer saving...")
        self.replay_buffer.save(filename)
        print("Buffer saved.")

    def load_replay_buffer(self, filename):
        self.replay_buffer.load(filename)
