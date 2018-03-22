from arm_env import ArmEnv
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

env = ArmEnv(size_x=4, size_y=3, cubes_cnt=4, episode_max_length=1000, finish_reward=200,
             action_minus_reward=-1, tower_target_size=3)

env.render()

def to_cat(a, n):
    return np.array([a if a == i else 0 for i in range(n)])


#hyperparams
units_1 = 64
units_output = 6
alpha = 0.05
gamma = 1
obs_space = len(env.reset())
act_space = len(env.ACTIONS)
num_steps = 100
num_episodes = 100

tf.reset_default_graph()

# graph
state = tf.placeholder('float32', shape=[None, obs_space], name="INPUT_STATE")
dense_1 = tf.layers.dense(state, units_1, kernel_initializer=tf.initializers.constant(1), name="DENSE1")
output = tf.layers.dense(dense_1, units_output, kernel_initializer=tf.initializers.constant(1), name="OUTPUT")
softmaxed_output = tf.nn.softmax(output)

# summary
tf.summary.histogram('Probabilities of actions', softmaxed_output)

# vars
returns = tf.placeholder('float32', name='RETURNS')
actions = tf.placeholder('int32', name="ACTIONS")

opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
wnl = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=output, name="WNL") #weigted negative likelihood
loss = -tf.reduce_mean(tf.multiply(wnl, returns))
# grads = opt.compute_gradients(loss)

'''
logits = tf.log(softmaxed_output)
loss = tf.reduce_sum(tf.matmul(returns, logits))
upd = opt.minimize(-loss, global_step=tf.train.get_global_step())
'''
upd = opt.minimize(loss)

# main frame
with tf.Session() as sess:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    train_writer = tf.summary.FileWriter('./logs', sess.graph)
    sess.run(tf.global_variables_initializer())
    for episode in range(num_episodes):
        cumulative_reward = 0
        current_state = np.array(env.reset())
        set_of_rewards = []
        set_of_actions = []
        set_of_states = []
        for step in range(num_steps):
            set_of_states.append(current_state)
            out, soft_out = sess.run([output, softmaxed_output], feed_dict={state: [current_state]})
            action = np.random.choice(len(env.ACTIONS), p=soft_out[0])
            new_state, reward, done, _ = env.step(action)
            set_of_rewards.append(reward * gamma ** step)
            #set_of_actions.append(to_cat(action, act_space))
            set_of_actions.append(action)
            current_state = new_state
            if done:
                env.render()
                print(episode, step)
                break

        # print(np.array(set_of_states).shape, np.array(set_of_actions).shape, np.array(set_of_rewards).shape)
        new_set_of_rewards = []
        for i in range(len(set_of_rewards)):
            new_set_of_rewards.append(np.sum(set_of_rewards[i:]))

        merge = tf.summary.merge_all()

        _, so, ls = sess.run([upd, softmaxed_output, loss], feed_dict={state: set_of_states,
                                                                       actions: set_of_actions,
                                                                       returns: [new_set_of_rewards]})

        #if episode % 5 == 0:
            #print("Episode: ", episode, "loss: ", ls)
            #print("So: ", episode, "loss: ", so)
            #print("set of actions: ", set_of_actions)
            #env.render()
