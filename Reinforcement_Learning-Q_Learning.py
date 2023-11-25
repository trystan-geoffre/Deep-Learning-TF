
import gym
import numpy as np

# Environment Setup
env = gym.make('FrozenLake-v1')

#print(env.observation_space.n)
#print(env.action_space.n)

# Initialization and Exploration
env.reset()
action = env.action_space.sample()
#new_state,reward,done,info = env.step(action)

#env.render()

# Q-Table Initialization
STATES = env.observation_space.n
ACTIONS = env.action_space.n

Q = np.zeros((STATES, ACTIONS))

# Parameters and Hyperparameters
EPISODES = 2000
MAX_STEPS = 100
LEARNING_RATE = 0.81
GAMMA = 0.96
RENDER = False
epsilon = 0.9

#if np.random.uniform(0,1)< epsilon:
#    action = env.action_space.sample()
#else:
#    action = np.argmax(Q[state,:])

#Q[state, action] = Q[sate, action] + LEARNING_RATE*(reward + GAMMA*np.max(Q[new_state, :]) - Q[state,action ])


# Q-Learning Loop
rewards = []
for episode in range(EPISODES):
    state = env.reset()
    for _ in range(MAX_STEPS):

        # Exploration and Exploitation
        if RENDER:
            env.render()

        if np.random.uniform(0,1)<epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Taking an Action and Updating Q-Table
        next_state, reward, done, _ = env.step(action)

        Q[state, action] = Q[state, action] + LEARNING_RATE*(reward + GAMMA *np.max(Q[next_state, :])- Q[state, action])

        state = next_state

        # Checking for Episode Termination
        if done:
            rewards.append(reward)
            epsilon -=0.001
            break

# Results
print(Q)
print(f"Average reward: {sum(rewards)/(len(rewards))}:")
