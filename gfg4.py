import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

env = gym.make('CartPole-v1')
obs_space = env.observation_space.shape[0]
act_space = env.action_space.n
gamma = 0.99
learning_rate = 0.01
num_episodes = 1000
batch_size = 64
class PolicyNetwork(tf.keras.Model):
    def __init__(self, hidden_units=128):
        super(PolicyNetwork, self).__init__()
        self.dense1 = layers.Dense(hidden_units, activation='relu')
        self.dense2 = layers.Dense(env.action_space.n, activation='softmax')  

    def call(self, state):
        x = self.dense1(state)
        return self.dense2(x)
policy = PolicyNetwork()
optimizer = tf.keras.optimizers.Adam(learning_rate)
def compute_returns(rewards, gamma):
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    return returns
def train_step(states, actions, returns):
    with tf.GradientTape() as tape:
        # Calculate the probability of each action taken
        action_probs = policy(states)
        action_indices = np.array(actions, dtype=np.int32)
        
        # Gather the probabilities for the actions taken
        action_log_probs = tf.math.log(tf.reduce_sum(action_probs * tf.one_hot(action_indices, env.action_space.n), axis=1))

        # Calculate the loss (negative log likelihood * returns)
        loss = -tf.reduce_mean(action_log_probs * returns)

    grads = tape.gradient(loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    states, actions, rewards = [], [], []

    while not done:
        state_input = np.array(state, dtype=np.float32).reshape(1, -1)
        probs = policy(state_input).numpy()[0]
        action = np.random.choice(act_space, p=probs)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state_input[0])
        actions.append(action)
        rewards.append(reward)
        state = next_state

    # After episode ends
    returns = compute_returns(rewards, gamma)
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-9)

    states_batch = np.vstack(states)
    train_step(states_batch, actions, returns)

    if episode % 100 == 0:
        print(f"Episode {episode}/{num_episodes}")
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    states, actions, rewards = [], [], []

    while not done:
        state_input = np.array(state, dtype=np.float32).reshape(1, -1)
        probs = policy(state_input).numpy()[0]
        action = np.random.choice(act_space, p=probs)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(state_input[0])
        actions.append(action)
        rewards.append(reward)
        state = next_state

    # After episode ends
    returns = compute_returns(rewards, gamma)
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-9)

    states_batch = np.vstack(states)
    train_step(states_batch, actions, returns)

    if episode % 100 == 0:
        print(f"Episode {episode}/{num_episodes}")
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    state_input = np.array(state, dtype=np.float32).reshape(1, -1)
    probs = policy(state_input).numpy()[0]
    action = np.argmax(probs)

    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    state = next_state

print(f"Test Total Reward: {total_reward}")