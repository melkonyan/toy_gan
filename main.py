import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


#seed = 100
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
    h2 = tf.nn.tanh(linear(h1, hidden_layer, 'd3'))
    h3 = tf.nn.sigmoid(linear(h2, 1, 'd4'))
    return h3


class GAN(object):

    scope_name = 'GAN'
    steps_num = 1000

    def prepare(self):
        self.create_models()
        self.create_training_vars()
        self.create_summary_entities()

    def create_models(self):
         with tf.variable_scope(self.scope_name):
            with tf.variable_scope('G'):
                self.noise = tf.placeholder(tf.float32, [None, 1], "z")
                self.gen = generator(self.noise)
            with tf.variable_scope('D') as scope:
                self.data = tf.placeholder(tf.float32, [None, 1], "d")
                self.dis_fake = discriminator(self.gen)
                scope.reuse_variables()
                self.dis_real = discriminator(self.data)

    def create_training_vars(self):
         with tf.variable_scope(self.scope_name):
            self.loss_gen = self.compute_loss_gen(self.dis_fake)
            self.loss_dis = self.compute_loss_dis(self.dis_real, self.dis_fake)
            optimizer = tf.train.RMSPropOptimizer(0.01)
            vars = tf.trainable_variables()
            self.d_params = [v for v in vars if v.name.startswith(self.scope_name+'/D/')]
            self.g_params = [v for v in vars if v.name.startswith(self.scope_name+'/G/')]
            self.opt_dis = optimizer.minimize(self.loss_dis, var_list=self.d_params)
            self.opt_gen = optimizer.minimize(self.loss_gen, var_list=self.g_params)

    def create_summary_entities(self):
        with tf.variable_scope(self.scope_name):
            tf.summary.scalar("generator loss", self.loss_gen, collections=[self.scope_name])
            tf.summary.scalar("discriminator loss", self.loss_dis, collections=[self.scope_name])


    def compute_loss_gen(self, dis_fake):
        """
        :param dis_fake: probabilities assigned by discriminator to the fake data
        """
        return tf.reduce_mean(-tf.log(dis_fake))

    def compute_loss_dis(self, dis_real, dis_fake):
        """
        :param dis_real: probabilities assigned by discriminator to the real data
        :param dis_fake: probabilities assigned by discriminator to the fake data
        """
        return tf.reduce_mean(-tf.log(dis_real)) + tf.reduce_mean(-tf.log(1-dis_fake))

    def dis_post_update(self, session):
        pass

    def create_session(self):
        session = tf.Session()
        init = tf.global_variables_initializer()
        session.run(init)
        self.merged_summary = tf.summary.merge_all(key=self.scope_name)
        self.summary_writer = tf.summary.FileWriter("logs/trial0", session.graph)
        return session

    def create_feed_dict(self, batch_size=1000):
        data_batch = gen_data(batch_size)
        noise_batch = gen_noise(batch_size)
        return {
            self.data: np.reshape(data_batch, (batch_size,1)),
            self.noise: np.reshape(noise_batch, (batch_size,1))
        }

    def run_one_training_step(self, step_num, tf_session, feed_dict):
        # Run one step of batch gradient descent
        tf_session.run(self.opt_gen, feed_dict)
        for i in range(10):
            tf_session.run(self.opt_dis, feed_dict)
            self.dis_post_update(tf_session)
        # Log loss values
        summary = tf_session.run(self.merged_summary, feed_dict)
        self.summary_writer.add_summary(summary, step_num)
        self.summary_writer.flush()
        if step_num % (self.steps_num / 10) == 0:
            l_g, l_d = tf_session.run([self.loss_gen, self.loss_dis], feed_dict)
            print("Gen loss: %g, dis loss: %g" % (l_g, l_d))

    def train(self, batch_size=1000, test_size=10000):
        #data_batch = gen_data(batch_size)
        #noise_batch = gen_noise(batch_size)
        with tf.variable_scope(self.scope_name):
            session = self.create_session()
            for step in range(self.steps_num):
                self.run_one_training_step(step, session, self.create_feed_dict(batch_size))
            t_noise = gen_noise(test_size)
            dec_size = 1000
            t_data = np.linspace(-10, 10, dec_size)
            feed_dict = {
                self.data: np.reshape(t_data, (dec_size, 1)),
                self.noise: np.reshape(t_noise, (test_size, 1))
            }
            dec_boundary = session.run(self.dis_real, feed_dict)
            print("Test generator loss: %g" % session.run(self.loss_gen, feed_dict))
            print("Test discriminator loss: %g" % session.run(self.loss_dis, feed_dict))
            return session.run(self.gen, feed_dict), np.transpose(np.column_stack((t_data, dec_boundary)))


class WGAN(GAN):

    scope_name='WGAN'

    def create_models(self):
        super().create_models()
        self.clip_by_value_operator = None

    def compute_loss_gen(self, dis_fake):
        return -tf.reduce_mean(dis_fake)

    def compute_loss_dis(self, dis_real, dis_fake):
        return -tf.reduce_mean(dis_real) + tf.reduce_mean(dis_fake)

    def dis_post_update(self, session):
        if self.clip_by_value_operator is None:
            self.clip_by_value_operator = [param.assign(tf.clip_by_value(param, -3, 3)) for param in self.d_params]
        session.run(self.clip_by_value_operator)


# Run discriminators of naive and
class GANsCrossValidation():

    def run(self):
        gan = GAN()
        wgan = WGAN()
        wgan.create_models()
        wgan.create_training_vars()
        gan.create_models()
        gan.create_training_vars()
        with tf.variable_scope(gan.scope_name+'/D') as scope:
            scope.reuse_variables()
            gan_dis_wgan_gen = discriminator(wgan.gen)
            gan_dis_wgan_real = discriminator(wgan.data)
        with tf.variable_scope(gan.scope_name):
            gan_cross_loss = gan.compute_loss_dis(gan_dis_wgan_real, gan_dis_wgan_gen)
            tf.summary.scalar("GAN discriminator vs WGAN generator", gan_cross_loss, collections=[gan.scope_name])
        gan.create_summary_entities()
        with tf.variable_scope(wgan.scope_name+'/D') as scope:
            scope.reuse_variables()
            wgan_dis_gan_gen = discriminator(gan.gen)
            wgan_dis_gan_real = discriminator(gan.data)
        with tf.variable_scope(wgan.scope_name):
            wgan_cross_loss = wgan.compute_loss_dis(wgan_dis_gan_real, wgan_dis_gan_gen)
            tf.summary.scalar("WGAN discriminator vs GAN generator", wgan_cross_loss, collections=[wgan.scope_name])
        wgan.create_summary_entities()
        with tf.variable_scope(gan.scope_name):
            gan_sess = gan.create_session()
        with tf.variable_scope(wgan.scope_name):
            wgan_sess = wgan.create_session()
        for step in range(100):
        #with tf.variable_scope(wgan.scope_name):
            wgan_feed_dict = wgan.create_feed_dict()
        #with tf.variable_scope(gan.scope_name):
            gan_feed_dict = gan.create_feed_dict()
            #Merge feed dictionaries
            feed_dict = wgan_feed_dict.copy()
            feed_dict.update(gan_feed_dict)
            #with tf.variable_scope(wgan.scope_name):
            wgan.run_one_training_step(step, wgan_sess, feed_dict)
            #with tf.variable_scope(gan.scope_name):
            gan.run_one_training_step(step, gan_sess, feed_dict)

GANsCrossValidation().run()
# real_data = gen_data(10000)
# gan = GAN()
# gan.prepare()
# fake_data, dec_boundary = gan.train()
# plt.hist(real_data, 100, histtype='step', normed=True, label='Real data')
# plt.hist(fake_data, 100, histtype='step', normed=True, label='Fake data')
# plt.plot(dec_boundary[0], dec_boundary[1], 'b', label='Decision boundary')
# plt.legend()
# plt.show()
#plt.savefig("images/batch.png")