import gym
from policy_gradient_layers import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np

from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
)

#env = gym.make("FrozenLake-v0")
env = gym.make('FrozenLakeNotSlippery-v0')

def to_cat(a, n):
    return np.array([1 if a == i else 0 for i in range(n)])

#env = gym.make('CartPole-v0')
#env = env.unwrapped

# Policy gradient has high variance, seed for reproducability
#env.seed(1)

print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)
#print("env.observation_space.high", env.observation_space.high)
#print("env.observation_space.low", env.observation_space.low)


RENDER_ENV = False
EPISODES = 5000
rewards = []
RENDER_REWARD_MIN = 50
succ = 0



if __name__ == "__main__":


    # Load checkpoint
    save_path = "./model/model.ckpt"
    load_path = None #"./output/weights/CartPole-v0-temp.ckpt"

    PG = PolicyGradient(
        n_x = env.observation_space.n,
        n_y = env.action_space.n,
        learning_rate=0.001,
        reward_decay=0.95
    )


    for episode in range(EPISODES):

        observation = to_cat(env.reset(), env.observation_space.n)
        episode_reward = 0

        while True:

            if RENDER_ENV: env.render()

            # 1. Choose an action based on observation
            action = PG.choose_action(observation)

            # 2. Take action in the environment
            observation_, reward, done, info = env.step(action)
            observation_ = to_cat(observation_, env.observation_space.n)

            # 3. Store transition for training
            PG.store_transition(observation, action, reward)

            if done:
                if reward != 0:
                    succ += reward
                episode_rewards_sum = sum(PG.episode_rewards)
                rewards.append(episode_rewards_sum)
                max_reward_so_far = np.amax(rewards)
                print("==========================================")
                print("Episode: ", episode)
                print("Reward: ", episode_rewards_sum)
                print("Max reward so far: ", max_reward_so_far)

                # 4. Train neural network
                discounted_episode_rewards_norm = PG.learn()

                # Render env if we get to rewards minimum
                if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = True


                break

            # Save new observation
            observation = observation_
    print(succ)