import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface as SOI
from tensorflow.contrib.distributions import kl_divergence as kl
import numpy as np
import gym
import time
import matplotlib.pyplot as plt
from arm_env import ArmEnv


class AC:
    def __init__(self, env, gamma=1, lr=0.001, num_episodes=1000, num_steps=200, KL_delta=10 ** (-4)):

        #self.env = ArmEnv(size_x=4, size_y=3, cubes_cnt=4, episode_max_length=2000, finish_reward=200,
        #                  action_minus_reward=0.0, tower_target_size=3)
        self.env = env
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
        self.batch_size = 5
        self.disc = 0
        self.log_file = open('./logs/logs_' + str(time.time()) + '.txt', 'w+')

        self.sess = tf.Session()

    def build_graph(self):

        '''
        =======
        BATCHES
        =======
        '''

        tf.reset_default_graph()

        # building graph:
        # dimensions:

        # S = [NUM_OF_STEPS, DIMENSION]
        # A = [NUM_OF_STEPS, 1]

        self.S = tf.placeholder('float64', shape=[None, len(self.env.reset())], name='S')
        self.A = tf.placeholder('int64', name='A')
        self.R = tf.placeholder('float64', name='R')
        self.build_actor()
        self.build_critic()


        #TODO: Redo action number -> one hot ---- NO, BECAUSE LOSS WILL NOT WORK


    '''
    ============================
    Actor = Policy Approxiamtion
    ============================
    '''
    def build_actor(self):
        self.inp = tf.layers.dense(
            self.S,
            10, # ??
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

        # Batch loss
        wnl = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.out, labels=self.A)
        ls = tf.reduce_mean(tf.multiply(self.R, wnl))
        self.opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(ls)


    '''
    ======================================
    Critic = Approximation of Q-function
    ======================================
    '''
    def build_critic(self):

        self.q_return = tf.placeholder('float32', name="Q-Return")  # sum of rewards on rest of traj
        self.traj_len = tf.placeholder('int32', name='Traj_Len')

        conc = tf.concat((self.S, tf.reshape([tf.cast(self.A, tf.float64)], shape=(self.traj_len, 1))), axis=1)
        self.q_inp = tf.layers.dense(
            conc,
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
        self.q_opt = tf.train.AdamOptimizer(0.001).minimize(self.q_loss)


    def roll_trajectory(self, episode):
        s = self.env.reset()
        self.trajectory = []
        self.total_rew = []

        for step in range(self.num_steps):
            output = self.sess.run([self.soft_out], feed_dict={self.S: [s]})
            probs = output[0][0]
            if not self.learn_flag:
                a = np.random.choice(self.env.action_space.n,
                                     p=[1. / self.env.action_space.n for _ in range(self.env.action_space.n)])
                #print("probs: ", probs, self.learn_flag, "action: ", a)
                self.log_file.write('probs: ' + str(probs) + '\n')
            else:
                a = np.random.choice(self.env.action_space.n, p=probs)
                #print("probs: ", probs, self.learn_flag, "action: ", a)
                self.log_file.write('probs: ' + str(probs) + ' step:' + str(step) + '\n')
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

        print('====================== end of episode {} ======================'.format(episode))

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
                self.roll_trajectory(episode)

                # Making learning sets
                S_traj = []
                A_traj = []
                R_traj = []

                for item in self.trajectory:
                    S_traj.append(item[0])
                    A_traj.append(item[1])
                    R_traj.append(item[2])


                disc = self.discount_and_norm_rewards(R_traj, gamma=1)

                #Learning Critic
                q_approximated, _ = self.sess.run([self.q_out, self.q_opt],
                                                  feed_dict={self.S: S_traj,
                                                             self.A: A_traj,
                                                             self.q_return: disc,
                                                             self.traj_len: len(R_traj)}
                                                  )

                #Learning Actor
                _ = self.sess.run([self.opt],
                                        feed_dict={self.S: S_traj,
                                                   self.A: A_traj,
                                                   self.R: q_approximated}
                                        )

                #Optimization Actor-Parameters
                #self.apply_trpo_SOI(traj_state, traj_action, q_approximated, soft, traj_reward, adv)

                if episode % self.slice == 0:
                    print("Episode: ", episode)
                    print("Successes to all: ", self.success_counter / self.slice)
                    self.success_episodes.append(episode)
                    self.successes.append(self.success_counter / self.slice)
                    self.success_counter = 0

            plt.plot(self.success_episodes, self.successes)
            plt.show()
            self.log_file.close()
            print(self.successes)

    @staticmethod
    def kl(p, q):
        return tf.reduce_sum(tf.multiply(p, tf.log(p / q)))

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


trpo = AC(ArmEnv(size_x=4, size_y=3, cubes_cnt=4, episode_max_length=2000, finish_reward=200,
                 action_minus_reward=0.0, tower_target_size=3), num_episodes=10000)
trpo.learn()
