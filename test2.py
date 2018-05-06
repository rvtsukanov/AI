import tensorflow as tf

tf.reset_default_graph()

S = tf.placeholder('float64', shape=[None, 4], name='S')
A = tf.placeholder('int64', name='A')
R = tf.placeholder('float64')


inp = tf.layers.dense(
    S,
    10,
    name="ACTOR_INPUT",
    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
    bias_initializer=tf.initializers.constant(0)
)

out = tf.layers.dense(
    inp,
    2,
    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
    bias_initializer=tf.initializers.constant(0)
)

soft_out = tf.nn.softmax(out)

r = tf.rank(S)
wns = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=A)
ls = tf.reduce_mean(tf.multiply(R, wns))

conc = tf.concat((S, tf.reshape([tf.cast(A, tf.float64)], (2, 1))), 1)

q_inp = tf.layers.dense(
    conc,
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


q_loss = tf.losses.mean_squared_error(q_out, R)
#q_opt = tf.train.AdamOptimizer(0.01).minimize(q_loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    # Assume that we have 5 steps in traj
    o, a, qo = sess.run([out, A, q_loss], feed_dict={S: [[0, 1, 0, 0],
                                                   [1, 1, 1, 1]],

                                               A: [0, 1],
                                               R: [200., 200.]})

    print(qo)

