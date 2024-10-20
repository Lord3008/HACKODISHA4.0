import numpy as np
import tensorflow as tf
import random
import pandas as pd

# Load the dataset
df = pd.read_csv('/mnt/data/df_transactions.csv')

# Extract relevant features for the state
# We use gas, gas_price, receipt_status, and risk_score as state variables
df['risk_score'] = np.random.uniform(0, 1, len(df))  # Simulate risk score as it's not present in the dataset
states = df[['gas', 'gas_price', 'receipt_status', 'risk_score']].values

# Environment parameters
num_paths = len(states)  # Each transaction can be treated as a path
state_size = states.shape[1]
action_size = num_paths

# Hyperparameters
learning_rate = 0.01
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 1000
batch_size = 32

# Reward function
def get_reward(state):
    gas, gas_price, receipt_status, risk_score = state
    reward = - (gas * 0.3 + gas_price * 0.2 + risk_score * 50)
    # Add a positive reward if the transaction is successful
    if receipt_status == 1:
        reward += 50
    return reward

# Build Q-network model
def build_model(state_size, action_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, input_dim=state_size, activation='relu'),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')
    return model

# Q-network and replay memory
q_network = build_model(state_size, action_size)
memory = []

# Store experiences in memory
def store_memory(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))
    if len(memory) > 2000:
        memory.pop(0)

# Epsilon-greedy action selection
def choose_action(state):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    q_values = q_network.predict(np.array([state]))
    return np.argmax(q_values[0])

# Training the Q-network
def train_q_network(batch_size=32):
    minibatch = random.sample(memory, min(batch_size, len(memory)))
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target += gamma * np.amax(q_network.predict(np.array([next_state]))[0])
        q_values = q_network.predict(np.array([state]))
        q_values[0][action] = target
        q_network.fit(np.array([state]), q_values, epochs=1, verbose=0)

# Training loop
for episode in range(num_episodes):
    # Randomly select a transaction as the initial state
    idx = random.randint(0, len(states) - 1)
    state = states[idx]
    total_reward = 0
    done = False

    while not done:
        action = choose_action(state)  # Choose an action (path/transaction)
        next_state_idx = random.randint(0, len(states) - 1)  # Random next transaction
        next_state = states[next_state_idx]

        reward = get_reward(state)  # Calculate reward
        done = random.uniform(0, 1) < 0.1  # Randomly end the episode

        store_memory(state, action, reward, next_state, done)  # Store the experience
        state = next_state
        total_reward += reward

        # Train Q-network from memory
        if len(memory) > batch_size:
            train_q_network(batch_size)

    # Decay epsilon for less exploration
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Print episode summary
    print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon}")

print("Training complete.")
