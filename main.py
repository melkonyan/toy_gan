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


def gen_noise(num_of_samples, range=8):
    return np.linspace(-range, range, num_of_samples) + np.random.random(num_of_samples) * 0.01


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


class GAN(object):

    def __init__(self):
        self._create_models()

    def _create_models(self):
        with tf.variable_scope('G'):
            self.noise = tf.placeholder(tf.float32, [None, 1], "z")
            self.gen = generator(self.noise)
        with tf.variable_scope('D') as scope:
            self.data = tf.placeholder(tf.float32, [None, 1], "d")
            self.dis_fake = discriminator(self.gen)
            scope.reuse_variables()
            self.dis_real = discriminator(self.data)

    def _loss_gen(self):
        return tf.reduce_mean(-tf.log(self.dis_fake))

    def _loss_dis(self):
        return tf.reduce_mean(-tf.log(self.dis_real)) + tf.reduce_mean(-tf.log(1-self.dis_fake))

    def _dis_post_update(self, session):
        pass

    def train(self, batch_size=1000, test_size=10000):
        loss_gen = self._loss_gen()
        loss_dis = self._loss_dis()
        optimizer = tf.train.RMSPropOptimizer(0.01)
        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]
        opt_dis = optimizer.minimize(loss_dis, var_list=self.d_params)
        opt_gen = optimizer.minimize(loss_gen, var_list=self.g_params)
        #data_batch = gen_data(batch_size)
        #noise_batch = gen_noise(batch_size)
        session = tf.Session()
        init = tf.global_variables_initializer()
        session.run(init)
        steps_num = 1000
        for step in range(steps_num):
            data_batch = gen_data(batch_size)
            noise_batch = gen_noise(batch_size)
            feed_dict = {
                self.data: np.reshape(data_batch, (batch_size,1)),
                self.noise: np.reshape(noise_batch, (batch_size,1))
            }
            session.run(opt_gen, feed_dict)
            session.run(opt_dis, feed_dict)
            self._dis_post_update(session)
            if step % (steps_num / 10) == 0:
                l_g, l_d = session.run([loss_gen, loss_dis], feed_dict)
                print("Gen loss: %g, dis loss: %g" % (l_g, l_d))
        t_noise = gen_noise(test_size)
        dec_size = 1000
        t_data = np.linspace(-10, 10, dec_size)
        feed_dict = {
            self.data: np.reshape(t_data, (dec_size, 1)),
            self.noise: np.reshape(t_noise, (test_size, 1))
        }
        dec_boundary = session.run(self.dis_real, feed_dict)
        print("Test generator loss: %g" % session.run(loss_gen, feed_dict))
        print("Test discriminator loss: %g" % session.run(loss_dis, feed_dict))
        return session.run(self.gen, feed_dict), np.transpose(np.column_stack((t_data, dec_boundary)))


class WGAN(GAN):


    def _create_models(self):
        super()._create_models()
        self.clip_by_value_operator = None

    def _loss_gen(self):
        return -tf.reduce_mean(self.dis_fake)

    def _loss_dis(self):
        return -tf.reduce_mean(self.dis_real) + tf.reduce_mean(self.dis_fake)

    def _dis_post_update(self, session):
        if self.clip_by_value_operator is None:
            self.clip_by_value_operator = [param.assign(tf.clip_by_value(param, -3, 3)) for param in self.d_params]
        session.run(self.clip_by_value_operator)

real_data = gen_data(10000)
fake_data, dec_boundary = WGAN().train()
plt.hist(real_data, 100, histtype='step', normed=True, label='Real data')
plt.hist(fake_data, 100, histtype='step', normed=True, label='Fake data')
plt.plot(dec_boundary[0], dec_boundary[1], 'b', label='Decision boundary')
plt.legend()
plt.show()
#plt.savefig("images/batch.png")