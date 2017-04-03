import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


def gen_data(num_of_samples):
    mu = 4.0
    sigma = 1.0
    samples = np.random.normal(mu, sigma, num_of_samples)
    return samples


def gen_noise(num_of_samples):
    return np.linspace(0, 1, num_of_samples) + np.random.random(num_of_samples) * 0.01


def linear(input, output_size, scope='linear'):
    input_size = input.shape[-1]
    with tf.variable_scope(scope):
        w = tf.get_variable("w", [input_size, output_size], tf.float32)
        b = tf.get_variable("b", [1, output_size], tf.float32)
    return tf.matmul(input, w) + b


def generator(noise, hidden_layer=10):
    h0 = tf.nn.softplus(linear(noise, hidden_layer, 'g1'))
    return linear(h0, 1, 'g2')


def discriminator(data, hidden_layer = 10):
    h0 = tf.nn.tanh(linear(data, hidden_layer, 'd1'))
    h1 = tf.nn.tanh(linear(h0, hidden_layer, 'd2'))
    #h2 = tf.nn.tanh(linear(h1, hidden_layer, 'd3'))
    h3 = tf.nn.sigmoid(linear(h1, 1, 'd4'))
    return h3


def gan(batch_size=1000, test_size=10000):
    N = batch_size
    with tf.variable_scope('G'):
        noise = tf.placeholder(tf.float32, [None, 1], "z")
        gen = generator(noise)
    with tf.variable_scope('D') as scope:
        data = tf.placeholder(tf.float32, [None, 1], "d")
        dis_fake = discriminator(gen)
        scope.reuse_variables()
        dis_real = discriminator(data)
    loss_gen = tf.reduce_mean(-tf.log(dis_fake))
    loss_dis = tf.reduce_mean(-tf.log(dis_real)) + tf.reduce_mean(-tf.log(1-dis_fake))
    optimizer = tf.train.RMSPropOptimizer(0.001)
    vars = tf.trainable_variables()
    d_params = [v for v in vars if v.name.startswith('D/')]
    g_params = [v for v in vars if v.name.startswith('G/')]
    opt_dis = optimizer.minimize(loss_dis, var_list=d_params)
    opt_gen = optimizer.minimize(loss_gen, var_list=g_params)
    # data_batch = gen_data(batch_size)
    # noise_batch = gen_noise(batch_size)
    with tf.Session() as session:
        init = tf.global_variables_initializer()
        session.run(init)
        steps_num = 1000
        for step in range(steps_num):
            data_batch = gen_data(batch_size)
            noise_batch = gen_noise(batch_size)
            feed_dict = {
                data: np.reshape(data_batch, (batch_size,1)),
                noise: np.reshape(noise_batch, (batch_size,1))
            }
            session.run(opt_gen, feed_dict)
            session.run(opt_dis, feed_dict)
            if step % (steps_num / 10) == 0:
                l_g, l_d = session.run([loss_gen, loss_dis], feed_dict)
                print("Gen loss: %g, dis loss: %g" % (l_g, l_d))
        t_noise = gen_noise(test_size)
        dec_size = 1000
        t_data = np.linspace(-10, 10, dec_size)
        feed_dict = {
            data: np.reshape(t_data, (dec_size, 1)),
            noise: np.reshape(t_noise, (test_size, 1))
        }
        dec_boundary = session.run(dis_real, feed_dict)
        print("Test generator loss: %g" % session.run(loss_gen, feed_dict))
        print("Test discriminator loss: %g" % session.run(loss_dis, feed_dict))
        return session.run(gen, feed_dict), np.transpose(np.column_stack((t_data, dec_boundary)))

real_data = gen_data(10000)
fake_data, dec_boundary = gan()
plt.hist(real_data, 100, histtype='step', normed=True, label='Real data')
plt.hist(fake_data, 100, histtype='step', normed=True, label='Fake data')
plt.plot(dec_boundary[0], dec_boundary[1], 'b', label='Decision boundary')
plt.legend()
#plt.show()
plt.savefig("images/mini_batch.png")