import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base
import os
np.random.seed(0)
tf.set_random_seed(0)

'''
    Version 3.0 - Nov15/2019
    Jorge R. Vergara
    jorgever@utem.cl
    Santiago, Chile
'''


class DataSet(object):
    def __init__(self,
                 data,
                 labels,
                 one_hot=False,
                 seed=None):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        if one_hot:
            vect_class, idx_class = np.unique(labels, return_inverse=True)
            labels = self.dense_to_one_hot(idx_class, len(vect_class))
            self.one_hot = one_hot

        self._num_examples = data.shape[0]
        self._num_features = data.shape[1]
        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def num_features(self):
        return self._num_features

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch

        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._data = self.data[perm0]
            self._labels = self.labels[perm0]

        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            data_rest_part = self._data[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._data = self.data[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._labels[start:end]

    def dense_to_one_hot(self, labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot


def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random.uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class Autoencoder(object):
    """ Autoencoder (AE) with an sklearn-like interface implemented using TensorFlow.

    References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
    """

    def __init__(self,
                 transfer_fct=tf.nn.relu,
                 learning_rate=0.001,
                 batch_size=100,
                 training_epochs=10,
                 display_step=5,
                 n_z=100,
                 n_hidden=[549, 110, 110, 549],
                 sess=None):

        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.display_step = display_step
        self.n_z = n_z
        self.n_hidden = n_hidden
        self.network_architecture = self._network_architecture()
        self.sess=sess

    def _start(self):
        # tf Graph input
        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.network_architecture["n_input"]])

        # Create autoencoder network
        self._create_network()
        # Define loss function 
        self._create_loss_optimizer()
        self.saver = tf.compat.v1.train.Saver()

        # Launch the session
        if self.sess == None:
            self.sess = tf.compat.v1.Session()

        # Initializing the tensor flow variables
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    def _network_architecture(self):

        network_architecture = \
            dict(n_hidden_recog_1=self.n_hidden[0],  # 1st layer encoder neurons
                 n_hidden_recog_2=self.n_hidden[1],  # 2nd layer encoder neurons
                 n_hidden_gener_1=self.n_hidden[2],  # 1st layer decoder neurons
                 n_hidden_gener_2=self.n_hidden[3],  # 2nd layer decoder neurons
                 n_input=None,  # SDSS data input
                 n_z=self.n_z)  # dimensionality of latent space
        return network_architecture

    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine latent space
        self.z = self._recognition_network(network_weights["weights_recog"],network_weights["biases_recog"])

        # Use generator
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1, n_hidden_gener_2,
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights

    def _recognition_network(self, weights, biases):
        # Generate encoder (recognition network), which
        # maps inputs onto a latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        z = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        return z

    def _generator_network(self, weights, biases):
        # Generate decoder (decoder network), which
        # maps points in latent space to data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        x_reconstr_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                                 biases['out_mean'])
        return x_reconstr_mean

    def _create_loss_optimizer(self):
        # Define loss and optimizer, minimize the squared error
        self.cost = tf.reduce_mean(tf.square(self.x - self.x_reconstr_mean))
        # Use ADAM optimizer
        self.optimizer = \
            tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: X})
        return cost

    def transform(self, X):
        """Transform data by mapping it into the latent space(Z). e.g. X_input -> Z"""

        nbt = self.batch_size
        n_samples = X.shape[0]
        p = list(divmod(n_samples, nbt))
        if p[1] > 0:
            X = np.vstack((X, np.random.rand(nbt - p[1], X.shape[1])))
            p[0] = p[0] + 1

        temp = list()
        for ii in range(p[0]):
            i1 = nbt * ii
            i2 = nbt * (ii + 1)
            temp.append(self.sess.run(self.z, feed_dict={self.x: X[i1:i2]}))
        Xt = np.vstack(temp)

        return Xt[:n_samples]

    def generate(self, z_mu):
        """ Generate data by sampling from latent space, e.g.  Z -> X_recons."""

        nbt = self.batch_size
        n_samples = z_mu.shape[0]
        p = list(divmod(n_samples, nbt))
        if p[1] > 0:
            z_mu = np.vstack((z_mu, np.random.rand(nbt - p[1], z_mu.shape[1])))
            p[0] = p[0] + 1

        temp = list()
        for ii in range(p[0]):
            i1 = nbt * ii
            i2 = nbt * (ii + 1)
            temp.append(self.sess.run(self.x_reconstr_mean, feed_dict={self.z: z_mu[i1:i2]}))
        z_mut = np.vstack(temp)

        return z_mut[:n_samples]

    def reconstruct(self, X):
        """ Use AE to reconstruct given data. e.g. X_orig -> X_recons"""
        nbt = self.batch_size
        n_samples = X.shape[0]
        p = list(divmod(n_samples, nbt))
        if p[1] > 0:
            X = np.vstack((X, np.random.rand(nbt - p[1], X.shape[1])))
            p[0] = p[0] + 1

        temp = list()
        for ii in range(p[0]):
            i1 = nbt * ii
            i2 = nbt * (ii + 1)
            temp.append(self.sess.run(self.x_reconstr_mean, feed_dict={self.x: X[i1:i2]}))
        Xt = np.vstack(temp)

        return Xt[:n_samples]

    def _np2tf(self, X, labels=None):
        n_samples = X.shape[0]
        if labels is None:
            labels = np.ones(n_samples, dtype=int)
        return base.Datasets(train=DataSet(X, labels), validation=[], test=[])

    def train(self, X, labels=None):

        n_samples = X.shape[0]
        DATA = self._np2tf(X, labels)

        self.network_architecture['n_input'] = X.shape[1]
        self._start()

        # Training cycle
        for epoch in range(self.training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / self.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, _ = DATA.train.next_batch(self.batch_size)

                # Fit training using batch data
                cost = self.partial_fit(batch_xs)
                # Compute average loss
                avg_cost += cost / n_samples * self.batch_size

            # Display logs per epoch step
            if epoch % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "cost=", "{:.9f}".format(avg_cost))
        print('Finished training')

    def save(self,nameFolder):
        if not os.path.exists('temp'):
            os.makedirs('temp')
        if os.path.exists(os.path.join('temp',nameFolder)):
            import datetime
            nameFolderOld = nameFolder
            nameFolder = r'%s__%s' %(nameFolder,datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
            print('=======================================================')
            print('The %s folder exists. The name folder was changed to %s' %(nameFolderOld,nameFolder))
            print('=======================================================')
        nameFolder = os.path.join('temp',nameFolder)
        os.makedirs(nameFolder)
        nameFile1 = os.path.join(nameFolder,'model.ckpt')
        save_path = self.saver.save(self.sess, nameFile1)
        nameFile2 = os.path.join(nameFolder,'param.npz')
        np.savez(nameFile2,n_input=self.network_architecture['n_input'],
                 n_z=self.n_z,
                 transfer_fct=self.transfer_fct.__name__,
                 learning_rate=self.learning_rate,
                 batch_size=self.batch_size,
                 training_epochs=self.training_epochs,
                 display_step=self.display_step,
                 n_hidden=self.n_hidden,
                 network_architecture=self.network_architecture)
        print("Model saved in folder: %s" % nameFolder)

    def restore(self,nameFolder):
        if not os.path.exists('temp'):
            print('Folder ''temp'' does not  exist. Failed to restore model.')
        else:
            nameFolder = os.path.join('temp',nameFolder) 
            if not os.path.exists(nameFolder):
                print('Folder %s does not  exist. Failed to restore model.' % (nameFolder))
            param = np.load(os.path.join(nameFolder,'param.npz'), allow_pickle=True)
            self.transfer_fct = eval('tf.nn.'+param['transfer_fct'].item())
            self.learning_rate = param['learning_rate']
            self.batch_size = param['batch_size']
            self.training_epochs = param['training_epochs']
            self.display_step = param['display_step']
            self.n_z = param['n_z']
            self.n_hidden = param['n_hidden']
            self.network_architecture = param['network_architecture'].item()
            self._start()
            self.saver.restore(self.sess, os.path.join(nameFolder,'model.ckpt'))
            print("Model restored.")

    def close(self):
        self.sess.close()
