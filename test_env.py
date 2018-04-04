import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import gym
import matplotlib.pyplot as plt


from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
)


def to_cat(a, n):
    return np.array([1 if a == i else 0 for i in range(n)])


def discount_and_norm_rewards(episode_rewards, gamma):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0
    for t in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[t]
        discounted_episode_rewards[t] = cumulative
    return discounted_episode_rewards


env = gym.make('FrozenLakeNotSlippery-v0')
s = env.reset()

tf.reset_default_graph()
state = tf.placeholder('float32', shape=[None, env.observation_space.n], name="STATE")
actions = tf.squeeze(tf.placeholder('int32', name="ACTIONS"))
q = tf.placeholder('float32', name="Q")

inp = tf.layers.dense(
    state,
    10,
    name="INPUT",
    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
    bias_initializer=tf.initializers.constant(0.1)
)

out = tf.layers.dense(
    inp,
    env.action_space.n,
    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
    bias_initializer=tf.initializers.constant(0.1)
)

soft_out = tf.nn.softmax(out)

nl = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=actions)
wnl = tf.multiply(nl, q)
loss = tf.reduce_mean(wnl)
opt = tf.train.AdamOptimizer(learning_rate=0.05).minimize(loss)

saver = tf.train.Saver()

num_episodes = 5000
num_steps = 200

successes = []
success_episodes = []
slice = 100
success_counter = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for episode in range(num_episodes):
        s = env.reset()
        trajectory = []
        total_rew = []
        #if episode % 5 == 0:
        #   print("Episode: ", episode)
        for step in range(num_steps):
            s = to_cat(s, env.observation_space.n).reshape(1, -1)
            output = sess.run([soft_out], feed_dict={state: s})
            probs = output[0][0]
            a = np.random.choice(env.action_space.n, p=probs)
            new_state, reward, done, _ = env.step(a)
            total_rew.append(reward)
            trajectory.append((s, a, reward)) #no problems?

            if done:
                if reward != 0:
                    env.render()
                    success_counter += 1

                break
            s = new_state
        #print("Probs: ", probs)

        for n, st in enumerate(trajectory):
            ss = st[0]
            aa = [int(st[1])]
            rr = st[2]
            qq = sum(total_rew[n:])
            #print(aa, ss, rr, qq)
            _, ls = sess.run([opt, loss], feed_dict={state: ss, actions: aa, q: qq})

        if episode % slice == 0:
            print("Episode: ", episode)
            print("Successes to all: ", success_counter/slice)
            success_episodes.append(episode)
            successes.append(success_counter/slice)
            success_counter = 0


plt.plot(success_episodes, successes)
plt.show()
print(successes)



