import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import gym
import matplotlib.pyplot as plt


def to_cat(a, n):
    return np.array([1 if a == i else 0 for i in range(n)])

env = gym.make('FrozenLake-v0')
s = env.reset()

tf.reset_default_graph()

state = tf.placeholder('float32', shape=[None, env.observation_space.n], name="STATE")
actions = tf.squeeze(tf.placeholder('int32', name="ACTIONS"))
q = tf.placeholder('float32', name="Q")

#inp = tf.layers.dense(state, 8, name="INPUT", kernel_initializer=tf.initializers.zeros)
#out = tf.layers.dense(inp, env.action_space.n, kernel_initializer=tf.initializers.zeros)

inp = tf.layers.dense(state, 8, name="INPUT", kernel_initializer=tf.zeros_initializer())
out = tf.layers.dense(inp, env.action_space.n, kernel_initializer=tf.zeros_initializer())

sh = tf.shape(out)
soft_out = tf.nn.softmax(out)

nl = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=actions) # logs = [0.111, 0.222]; actions = 3
wnl = tf.multiply(nl, q)
loss = tf.reduce_mean(wnl)
opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)
upd = opt.minimize(-loss)

saver = tf.train.Saver()

num_episodes = 10000
num_steps = 200

successes = []
success_episodes = []
slice = 100
success_counter = 0

with tf.Session() as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(tf.global_variables_initializer())
    for episode in range(num_episodes):
        s = env.reset()
        trajectory = []
        total_rew = []
        #if episode % 5 == 0:
        #   print("Episode: ", episode)
        for step in range(num_steps):
            s = to_cat(s, env.observation_space.n).reshape(1, -1)
            #print(s)
            output = sess.run([soft_out], feed_dict={state: s})
            #print(output)
            probs = output[0][0]
            a = np.random.choice(env.action_space.n, p=probs)
            new_state, reward, done, _ = env.step(a)
            total_rew.append(reward)
            trajectory.append((s, a, reward))
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
            _, ls = sess.run([upd, loss], feed_dict={state: ss, actions: aa, q: qq})
        #save_path = saver.save(sess, "./models/model.ckpt")
        #if episode  == 0:
            #print("Loss: ", ls)
        if episode % slice == 0:
            print("Episode: ", episode)
            print("Successes to all: ", success_counter/slice)
            success_episodes.append(episode)
            successes.append(success_counter/slice)
            success_counter = 0


plt.plot(success_episodes, successes)
plt.show()
print(successes)


'''
from random import randint

dims = 8
pos = randint(0, dims - 1)

logits = tf.random_uniform([dims], maxval=3, dtype=tf.float32)
labels = tf.one_hot(pos, dims)

res1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
res2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.constant(pos))

with tf.Session() as sess:
    a, b, logs, labs = sess.run([res1, res2, logits, labels])
    print(a, b, logs, labs)
    print(a == b)
'''



