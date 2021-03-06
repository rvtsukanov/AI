import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface as SOI
from tensorflow.contrib.distributions import kl_divergence as kl
import numpy as np
import gym
import time
import matplotlib.pyplot as plt
from arm_env import ArmEnv


class TRPO():
    def __init__(self, env, gamma=1, lr=0.01, num_episodes=1000, num_steps=200, KL_delta=10 ** (-4)):

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
        self.disc = 0
        self.log_file = open('logs_' + str(time.time()) + '.txt', 'w+')

        self.sess = tf.Session()

    def build_graph(self):
        tf.reset_default_graph()
        self.state = tf.placeholder('float32', shape=[None, len(self.env.reset())], name="STATE")
        self.actions = tf.squeeze(tf.placeholder('int32', name="ACTIONS"))
        self.q_estimation = tf.placeholder('float32', name="Q-EST")
        self.build_actor()
        self.build_critic()
        self.build_trpo_tf()
        #self.build_trpo_SOI()


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

    def build_trpo_SOI(self):
        # I use index _k to fix variables in graph
        # Fixed probs on k-th iteration
        self.soft_out_k = tf.placeholder('float32', name="SOFTOUT_K")

        # Fixed advantage on k-th iteration
        self.A_k = tf.placeholder('float32', name="A_K")

        # Number of steps to estimate expectation
        self.N = tf.placeholder('float32', name="number")

        # Advantage function = emperical_return - baseline
        self.A = self.q_return - self.q_out

        # Choosing particular action "actions" and multiply by A_k
        #here was a mistake -> A instead of A_k
        trpo_obj = -tf.reduce_mean(self.A_k * tf.gather(tf.exp(self.soft_out - self.soft_out_k), self.actions))

        # KL(soft_out_k, soft_out) should be less than KL_delta
        constraints = [(-self.kl(self.soft_out_k, self.soft_out) + self.KL_delta)]

        # Use ScipyOptimiztationInterface (SOI) to solve optimization task with constrains
        self.trpo_opt = SOI(trpo_obj,
                            method='SLSQP',
                            inequalities=constraints,
                            options={'maxiter': 3})

    def apply_trpo_SOI(self, s, a, q_app, soft, r, adv):

        #Use trajectory s -> a -> r to optimize policy in Trust-Region interval
        feed_dict = [[self.state, [s]],
                     [self.soft_out_k, [soft]],
                     [self.actions, [a]],
                     [self.q_return, [r]],
                     [self.q_out, q_app],
                     [self.A_k, adv]]

        self.trpo_opt.minimize(self.sess, feed_dict=feed_dict)



    def build_trpo_tf(self):

        self.beta = tf.placeholder('float32')
        self.eta = tf.placeholder('float32')
        self.learn_rate = tf.placeholder('float32')
        self.learn_rate_value = 0.001

        self.soft_out_k = tf.placeholder('float32', name="SOFTOUT_K")
        # Fixed advantage on k-th iteration
        self.A_k = tf.placeholder('float32', name="A_K")
        self.A = self.q_return - self.q_out #?
        self.D_KL = self.kl(self.soft_out, self.soft_out_k)

        trpo_loss_1 = -tf.reduce_mean(self.A_k * tf.exp(self.soft_out - self.soft_out_k))
        trpo_loss_2 = self.beta * self.D_KL
        trpo_loss_3 = self.eta * tf.square(tf.maximum(0.0, self.KL_delta - 2 * self.D_KL))

        trpo_total_loss = trpo_loss_1 + trpo_loss_2 + trpo_loss_3
        self.trpo_opt = tf.train.AdamOptimizer(self.learn_rate).minimize(trpo_total_loss)



    def apply_trpo_tf(self, old_policy, advantage, state, actions, num_steps):
        beta = 0.5
        eta = 0.5
        DKL = 0.01
        for i in range(num_steps):
            if DKL > 2 * self.KL_delta:
                beta *= 1.5
                if beta > 30:
                    self.learn_rate_value /= 1.5
            elif DKL < 0.5 * self.KL_delta:
                beta /= 1.05
                if beta < 1./30:
                    self.learn_rate_value *= 1.5

            _, DKL = self.sess.run([self.trpo_opt, self.D_KL], feed_dict={self.A_k: advantage,
                                                          self.soft_out_k: old_policy,
                                                          self.actions: actions,
                                                          self.state: [state],
                                                          self.beta: beta,
                                                          self.eta: eta,
                                                          self.learn_rate: self.learn_rate_value})


    def roll_trajectory(self, episode):
        s = self.env.reset()
        self.trajectory = []
        self.total_rew = []

        for step in range(self.num_steps):
            output = self.sess.run([self.soft_out], feed_dict={self.state: [s]})
            probs = output[0][0]
            if not self.learn_flag:
                a = np.random.choice(self.env.action_space.n,
                                     p=[1. / self.env.action_space.n for _ in range(self.env.action_space.n)])
                #print("probs: ", probs, self.learn_flag, "action: ", a)
                self.log_file.write('probs: ' + str(probs) + '\n')
            else:
                a = np.random.choice(self.env.action_space.n, p=probs)
                #print("probs: ", probs, self.learn_flag, "action: ", a)
                self.log_file.write('probs: ' + str(probs) + '\n')
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
                    _, soft, adv = self.sess.run([self.opt, self.soft_out, self.A],
                                            feed_dict={self.state: [traj_state],
                                                       self.actions: [traj_action],
                                                       self.q_estimation: q_approximated,
                                                       self.q_return: traj_reward}
                                            )

                    #Optimization Actor-Parameters
                    #self.apply_trpo_SOI(traj_state, traj_action, q_approximated, soft, traj_reward, adv)
                    if episode > 500:
                        self.apply_trpo_tf(soft, adv, traj_state, traj_action, 20)

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


#trpo = TRPO(num_episodes=10000)
#trpo.learn()
