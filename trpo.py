import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface as SOI
import numpy as np
import gym
import matplotlib.pyplot as plt
from arm_env import ArmEnv

KL_delta = 10 ** (-4)


def kl_num(p, q):
    return np.sum(np.multiply(p, np.log(p/q)))


def kl(p, q):
    return tf.reduce_sum(tf.multiply(p, tf.log(p/q)))


def to_cat(a, n):
    return np.array([1 if a == i else 0 for i in range(n)])


def discount_and_norm_rewards(episode_rewards, gamma):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0
    for t in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[t]
        discounted_episode_rewards[t] = cumulative
    return discounted_episode_rewards


env = ArmEnv(size_x=4, size_y=3, cubes_cnt=4, episode_max_length=2000, finish_reward=200,
             action_minus_reward=0.0, tower_target_size=3)

s = env.reset()
obs_len = len(s)


tf.reset_default_graph()


state = tf.placeholder('float32', shape=[None, obs_len], name="STATE")
actions = tf.squeeze(tf.placeholder('int32', name="ACTIONS"))


'''
============================
Actor = Policy Approxiamtion
============================
'''
q_estimation = tf.placeholder('float32', name="Q-Estimation")

inp = tf.layers.dense(
    state,
    10,
    name="INPUT",
    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
    bias_initializer=tf.initializers.constant(0)
)

out = tf.layers.dense(
    inp,
    env.action_space.n,
    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
    bias_initializer=tf.initializers.constant(0)
)

soft_out = tf.nn.softmax(out)

nl = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=actions)
wnl = tf.multiply(nl, q_estimation)
loss = tf.reduce_mean(wnl)
opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

#########################


'''
=======================
Critic = Approximation of Q-function
=======================
'''
q_return = tf.placeholder('float32', name="Q-Return")  # sum of rewards on rest of traj
q_inp = tf.layers.dense(
    tf.concat(state, actions),
    10,
    name="Q-input",
    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
    bias_initializer=tf.initializers.constant(0)
)

q_out = tf.layers.dense(
    q_inp,
    1,
    name="Q-output",
    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
    bias_initializer=tf.initializers.constant(0)
)

q_loss = tf.losses.mean_squared_error(q_out, q_return)
q_opt = tf.train.AdamOptimizer(0.01).minimize(q_loss)

'''
Possible to use value-function to estimate advantage
=====================================
Value-function approximation (linear)
=====================================
'''
'''
value = tf.layers.dense(
    state,
    1,
    name="Value-Function",
    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
    bias_initializer=tf.initializers.constant(0)
)
#approximate with one-layer NN (linear combination of states)

v_loss = tf.losses.mean_squared_error(value, q_return)
v_opt = tf.train.AdamOptimizer(0.01).minimize(v_loss)
'''


# I use index _k to fix variables in graph

# Fixed probs on k-th iteration
soft_out_k = tf.placeholder('float32', name="SOFTOUT_K")

# Fixed advantage on k-th iteration
A_k = tf.placeholder('float32', name="A_K")

# Number of steps to estimate expectation
N = tf.placeholder('float32', name="number")

# Advantage function = emperical_return - baseline
A = q_return - q_out


cumulative_trpo_obj = 0

# Choosing particular action "actions" and multiply by A_k
trpo_obj = (tf.gather(tf.squeeze(soft_out), actions)/
                          tf.gather(tf.squeeze(soft_out_k), actions) * A_k)
cumulative_trpo_obj += trpo_obj

# KL(soft_out_k, soft_out) should be less than KL_delta
constraints = [(-kl(soft_out_k, soft_out) + KL_delta)]


#Use ScipyOptimiztationInterface (SOI) to solve optimization task with constrains
trpo_opt = SOI(-1./N * cumulative_trpo_obj,
               inequalities=constraints,
               method='SLSQP',
               options={'maxiter':1}) #it is not enough!
'''
=======================
HyperParams
=======================
'''
num_episodes = 10000
num_steps = 200
trpo_steps = 5
gamma = 0.9

successes = []
success_episodes = []
slice = 10
success_counter = 0
learn_flag = False

'''
=======================
Main
=======================
'''
try:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for episode in range(num_episodes):
            s = env.reset()
            trajectory = []
            total_rew = []
            for step in range(num_steps):
                output = sess.run([soft_out], feed_dict={state: [s]})
                probs = output[0][0]
                if not learn_flag:
                    a = np.random.choice(env.action_space.n,
                                         p=[1./env.action_space.n for i in range(env.action_space.n)])
                    print("probs: ", probs, learn_flag, "action: ", a)
                else:
                    a = np.random.choice(env.action_space.n, p=probs)
                    print("probs: ", probs, learn_flag)
                new_state, reward, done, _ = env.step(a)
                total_rew.append(reward)
                trajectory.append((s, a, reward))

                if done:
                    if reward != 0:
                        learn_flag = True
                        env.render()
                        print(reward)
                        success_counter += 1

                    break
                s = new_state
            disc = discount_and_norm_rewards(total_rew, gamma)

            if learn_flag:
                for n, st in enumerate(trajectory):
                    ss = st[0]
                    aa = int(st[1])
                    qq = disc[n]

                    #Learning Critic
                    q_approximated, _ = sess.run([q_out, q_opt],
                                                         feed_dict={state: [ss],
                                                                    actions: [aa],
                                                                    q_return: qq}
                                                         )
                    #Learning Actor
                    _, soft, adv = sess.run([opt, soft_out, A],
                                            feed_dict={state: [ss],
                                                       actions: [aa],
                                                       q_estimation: q_approximated,
                                                       q_return: qq}
                                            )
                    #Skip 10 episodes before TRPO to avoid noise in estimting advantge function
                    if episode > 10:
                        feed_dict = [[state, [ss]], [soft_out_k, [soft]], [A_k, [adv]], [actions, [aa]], [N, [n+episode+1]]]
                        #Calling optimizer
                        trpo_opt.minimize(sess, feed_dict=feed_dict)
                        print(soft)
                        print("adv: ", adv, " = ", qq, " - ", q_approximated)

                if episode % slice == 0:
                    print("Episode: ", episode)
                    print("Successes to all: ", success_counter/slice)
                    success_episodes.append(episode)
                    successes.append(success_counter/slice)
                    success_counter = 0




    plt.plot(success_episodes, successes)
    plt.show()
    print(successes)
except KeyboardInterrupt:
    plt.plot(success_episodes, successes)
    plt.show()
    raise




