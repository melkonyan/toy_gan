import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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
    h0 = tf.nn.relu(linear(noise, hidden_layer, 'g1'))
    return linear(h0, 1, 'g2')


def discriminator(data, hidden_layer = 10):
    h0 = tf.nn.relu(linear(data, hidden_layer, 'd1'))
    h1 = tf.nn.sigmoid(linear(h0, 1, 'd2'))
    return h1


def gan(noise, data):
    N = noise.shape[0]
    with tf.variable_scope('G'):
        noise = tf.constant(noise, tf.float32, [N, 1])
        gen = generator(noise)
    with tf.variable_scope('D') as scope:
        data = tf.constant(data, tf.float32, [N, 1])
        dis_fake = discriminator(gen)
        scope.reuse_variables()
        dis_real = discriminator(data)
    loss_gen = tf.reduce_mean(-tf.log(dis_fake))
    loss_dis = tf.reduce_mean(-tf.log(dis_real)-tf.log(1-dis_fake))
    optimizer = tf.train.AdamOptimizer()
    vars = tf.trainable_variables()
    d_params = [v for v in vars if v.name.startswith('D/')]
    g_params = [v for v in vars if v.name.startswith('G/')]
    opt_dis = optimizer.minimize(loss_dis, var_list=d_params)
    opt_gen = optimizer.minimize(loss_gen, var_list=g_params)
    with tf.Session() as session:
        init = tf.global_variables_initializer()
        session.run(init)
        steps_num = 1000
        for step in range(steps_num):
            session.run(opt_gen)
            session.run(opt_dis)
            if step % (steps_num / 10) == 0:
                l_g, l_d = session.run([loss_gen, loss_dis])
                print("Gen loss: %g, dis loss: %g" % (l_g, l_d))
        return session.run(gen)

real_data = gen_data(10000)
noise = gen_noise(10000)
print(noise.shape)
fake_data = gan(noise, real_data)
plt.hist(real_data, 100, histtype='step', normed=True, label='Real data')
plt.hist(fake_data, 100, histtype='step', normed=True, label='Fake data')
plt.savefig("images/batch.png")