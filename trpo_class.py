import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface as SOI
import numpy as np
import gym
import matplotlib.pyplot as plt
from arm_env import ArmEnv


class TRPO():
    def __init__(self, gamma=0.99, lr=0.01, num_episodes=1000, num_steps=200, KL_delta=10 ** (-4)):
        self.env = ArmEnv(size_x=4, size_y=3, cubes_cnt=4, episode_max_length=2000, finish_reward=200,
                          action_minus_reward=0.0, tower_target_size=3)
        self.gamma = gamma
        self.lr = lr
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.KL_delta = KL_delta
        self.success_counter = 0
        self.build_graph()
        self.obs_len = len(self.env.reset())
        self.trajectory = []
        self.total_rew = []
        self.disc = 0
        self.sess = tf.Session()

    def build_graph(self):
        tf.reset_default_graph()
        self.state = tf.placeholder('float32', shape=[None, len(self.env.reset())], name="STATE")
        self.actions = tf.squeeze(tf.placeholder('int32', name="ACTIONS"))
        self.q_estimation = tf.placeholder('float32', name="Q-EST")
        self.build_actor()
        self.build_critic()

    '''
    ============================
    Actor = Policy Approxiamtion
    ============================
    '''
    def build_actor(self):
        self.inp = tf.layers.dense(
            self.state,
            10,
            name="ACTOR_INPUT",
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
            bias_initializer=tf.initializers.constant(0)
        )

        self.out = tf.layers.dense(
            self.inp,
            self.env.action_space.n,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
            bias_initializer=tf.initializers.constant(0)
        )

        self.soft_out = tf.nn.softmax(self.out)
        nl = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.out, labels=self.actions)
        wnl = tf.multiply(nl, self.q_estimation)
        self.loss = tf.reduce_mean(wnl)
        self.opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    '''
    ======================================
    Critic = Approximation of Q-function
    ======================================
    '''
    def build_critic(self):
        self.q_return = tf.placeholder('float32', name="Q-Return")  # sum of rewards on rest of traj
        self.q_inp = tf.layers.dense(
            tf.concat(self.state, self.actions),
            10,
            name="Q-input",
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
            bias_initializer=tf.initializers.constant(0)
        )

        self.q_out = tf.layers.dense(
            self.q_inp,
            1,
            name="Q-output",
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
            bias_initializer=tf.initializers.constant(0)
        )

        self.q_loss = tf.losses.mean_squared_error(self.q_out, self.q_return)
        self.q_opt = tf.train.AdamOptimizer(0.01).minimize(self.q_loss)

    def build_trpo(self):
        # I use index _k to fix variables in graph
        # Fixed probs on k-th iteration
        self.soft_out_k = tf.placeholder('float32', name="SOFTOUT_K")

        # Fixed advantage on k-th iteration
        self.A_k = tf.placeholder('float32', name="A_K")

        # Number of steps to estimate expectation
        self.N = tf.placeholder('float32', name="number")

        # Advantage function = emperical_return - baseline
        self.A = self.q_return - self.q_out
        self.cumulative_trpo_obj = 0

        # Choosing particular action "actions" and multiply by A_k
        trpo_obj = (tf.gather(tf.squeeze(self.soft_out), self.actions) /
                    tf.gather(tf.squeeze(self.soft_out_k), self.actions) * self.A_k)
        self.cumulative_trpo_obj += trpo_obj

        # KL(soft_out_k, soft_out) should be less than KL_delta
        constraints = [(-self.kl(self.soft_out_k, self.soft_out) + self.KL_delta)]

        # Use ScipyOptimiztationInterface (SOI) to solve optimization task with constrains
        self.trpo_opt = SOI(-1. / self.N * self.cumulative_trpo_obj,
                            inequalities=constraints,
                            method='SLSQP',
                            options={'maxiter': 1})

    def roll_trajectory(self):
        s = self.env.reset()
        self.trajectory = []
        self.total_rew = []

        for step in range(self.num_steps):
            output = self.sess.run([self.soft_out], feed_dict={self.state: [s]})
            probs = output[0][0]
            if not self.learn_flag:
                a = np.random.choice(self.env.action_space.n,
                                     p=[1. / self.env.action_space.n for _ in range(self.env.action_space.n)])
                print("probs: ", probs, self.learn_flag, "action: ", a)
            else:
                a = np.random.choice(self.env.action_space.n, p=probs)
                print("probs: ", probs, self.learn_flag, "action: ", a)
            new_state, reward, done, _ = self.env.step(a)
            self.total_rew.append(reward)
            self.trajectory.append((s, a, reward))

            if done:
                if reward != 0:
                    self.env.render()
                    self.learn_flag = True
                    print(reward)
                    self.success_counter += 1
                return
            s = new_state

        print('======================')

    def learn(self):
        self.learn_flag = False
        with self.sess:
            self.sess.run(tf.global_variables_initializer())

            # Calculate metrics
            self.successes = []
            self.success_episodes = []
            self.slice = 10
            self.success_counter = 0

            for episode in range(self.num_episodes):
                self.roll_trajectory()
                disc = self.discount_and_norm_rewards(self.total_rew, self.gamma)
                for n, st in enumerate(self.trajectory):
                    traj_state = st[0]
                    traj_action = int(st[1])
                    traj_reward = disc[n]

                    #Learning Critic
                    q_approximated, _ = self.sess.run([self.q_out, self.q_opt],
                                                      feed_dict={self.state: [traj_state],
                                                                 self.actions: [traj_action],
                                                                 self.q_return: traj_reward}
                                                      )
                    #Learning Actor
                    _, soft = self.sess.run([self.opt, self.soft_out],
                                            feed_dict={self.state: [traj_state],
                                                       self.actions: [traj_action],
                                                       self.q_estimation: q_approximated,
                                                       self.q_return: traj_reward}
                                            )
                if episode % self.slice == 0:
                    print("Episode: ", episode)
                    print("Successes to all: ", self.success_counter / self.slice)
                    self.success_episodes.append(episode)
                    self.successes.append(self.success_counter / self.slice)
                    self.success_counter = 0

            plt.plot(self.success_episodes, self.successes)
            plt.show()
            print(self.successes)

    @staticmethod
    def kl_num(p, q):
        return np.sum(np.multiply(p, np.log(p/q)))

    @staticmethod
    def to_cat(a, n):
        return np.array([1 if a == i else 0 for i in range(n)])

    @staticmethod
    def discount_and_norm_rewards(episode_rewards, gamma):
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0
        for t in reversed(range(len(episode_rewards))):
            cumulative = cumulative * gamma + episode_rewards[t]
            discounted_episode_rewards[t] = cumulative
        return discounted_episode_rewards




trpo = TRPO(num_episodes=1000)
trpo.learn()
