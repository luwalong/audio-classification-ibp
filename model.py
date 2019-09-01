import tensorflow as tf
import numpy as np
import os, pathlib
from tqdm import tqdm
from tabulate import tabulate

from util import *
from interval_bounds import IntervalOps

FLAGS = tf.app.flags.FLAGS
io = IntervalOps()

class Model():
    def __init__(self):
        assert FLAGS.dataset in ['fsdd'], \
                f'Dataset:{FLAGS.dataset} is not supported for current version'

        if FLAGS.dataset == 'fsdd':
            self.n_digits = 10
            self.samp_freq = 8000

        self.lr = FLAGS.learning_rate
        #self.batch_size = FLAGS.batch_size
        self.train_epochs = FLAGS.train_epochs

        self.preemph_coeff = FLAGS.preemph_coeff
        self.n_cep = FLAGS.n_cep
        self.n_filt = FLAGS.n_filt
        self.n_hidden_dim = FLAGS.n_hidden_dim
        self.frame_size = FLAGS.frame_size
        self.frame_step = FLAGS.frame_step

        self.graph = self.build_graph()
        self.model_dir = os.getcwd() + '/models/' + FLAGS.model_name


    def LSTMCell_con(self, input, h_, c_, kernel, bias, forget_bias=1.0):
        mtx = tf.matmul(tf.concat([input, h_], 1), kernel) + bias
        i, j, f, o = tf.split(mtx, 4, 1)

        c = tf.add(
                tf.multiply(tf.sigmoid(tf.add(f, forget_bias)), c_),
                tf.multiply(tf.sigmoid(i), tf.tanh(j)))
        h = tf.multiply(tf.sigmoid(o), tf.tanh(c))
        return h, c


    def LSTMCell_ibp(self, input, h_, c_, kernel, bias, forget_bias=1.0):
        mtx = io.affine(io.concat([input, h_], 1), kernel, bias)
        i, j, f, o = io.split(mtx, 4, 1)

        c = io.add(
                io.multiply(io.sigmoid(io.add(f, forget_bias)), c_),
                io.multiply(io.sigmoid(i), io.tanh(j)))
        h = io.multiply(io.sigmoid(o), io.tanh(c))
        return h, c


    def build_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            # Input
            self.batch_size = tf.placeholder(dtype=tf.int32,
                    shape=[],
                    name='batch_size')
            self.signal_con = tf.placeholder(dtype=tf.float32,
                    shape=[None, None],
                    name='signal_con')
            self.eps = tf.placeholder(dtype=tf.float32,
                    shape=[],
                    name='epsilon')
            self.kappa = tf.placeholder(dtype=tf.float32,
                    shape=[],
                    name='kappa')
            self.labels = tf.placeholder(dtype=tf.int32,
                    shape=[None],
                    name='labels')

            # Variables & Initializer
            init = tf.contrib.layers.xavier_initializer()
            b1 = tf.get_variable('b1', [self.n_hidden_dim], tf.float32, init)
            W1 = tf.get_variable('W1', [self.n_filt, self.n_hidden_dim], 
                    tf.float32, init)
            kernel = tf.get_variable('lstm/kernel',
                    [self.n_hidden_dim*2, self.n_hidden_dim*4], tf.float32, init)
            bias = tf.get_variable('lstm/bias', [self.n_hidden_dim*4],
                    tf.float32, init)
            b2 = tf.get_variable('b2', [self.n_digits], tf.float32, init)
            W2 = tf.get_variable('W2', [self.n_hidden_dim, self.n_digits],
                    tf.float32, init)

            # Alias & Constants
            N = self.frame_size
            s = self.frame_step
            M = ConstantMatrices(N, s, self.preemph_coeff, 
                    self.n_filt, self.n_cep, self.samp_freq)

            # Pre-processing
            with tf.name_scope('preprocessing') as meta_scope:
                # Concrete Model
                with tf.name_scope('con') as domain_scope:
                    with tf.name_scope('window') as process_scope:
                        windowed_con = overlapping_windows(self.signal_con,
                                size=self.frame_size,
                                step=self.frame_step,
                                batch=self.batch_size)
                        windowed_tr_con = tf.transpose(windowed_con, [1, 0, 2])
                        windowed_lin_con = tf.reshape(windowed_tr_con, [-1, N])

                    with tf.name_scope('fft') as process_scope:
                        fft_comb_con = tf.square(
                                tf.matmul(windowed_lin_con, M.aux_mtx))
                        ffted_con = tf.matmul(fft_comb_con, M.fft_add_mtx)
        
                    with tf.name_scope('fb') as process_scope:
                        feat_con = tf.log(tf.matmul(ffted_con, M.fb)+1e-10)

                    with tf.name_scope('dct') as process_scope:
                        layer1_feed_con = tf.matmul(feat_con, M.dct_lift)
                
                # Interval Bound Model
                with tf.name_scope('ibp') as domain_scope:
                    # Stacking frames
                    with tf.name_scope('window') as process_scope:
                        windowed_ibp = make_interval_bounds(windowed_con,
                                self.eps)
                        windowed_tr_ibp = io.transpose(windowed_ibp, [1, 0, 2])
                        windowed_lin_ibp = io.reshape(windowed_tr_ibp, [-1, N])

                    with tf.name_scope('fft') as process_scope:
                        fft_comb_ibp = io.square(
                                io.affine(windowed_lin_ibp, M.aux_mtx))
                        ffted_ibp = io.affine(fft_comb_ibp, M.fft_add_mtx)
        
                    with tf.name_scope('fb') as process_scope:
                        feat_ibp = io.log(io.affine(ffted_ibp, M.fb)+1e-10)

                    with tf.name_scope('dct') as process_scope:
                        layer1_feed_ibp = io.affine(feat_ibp, M.dct_lift)

            # Prediction
            with tf.name_scope('prediction') as meta_scope:
                # In this scope, we do not separate the domain_scope since
                # it is better to calculate both simultaneously in LSTM layer.
                with tf.name_scope('fc1') as layer_scope:
                    layer1_con = tf.nn.relu(
                            tf.add(tf.matmul(layer1_feed_con, W1), b1))
                    layer1_ibp = io.relu(
                            io.affine(layer1_feed_ibp, W1, b1))

                with tf.name_scope('lstm') as layer_scope:
                    lstm_feed_con = tf.reshape(layer1_con,
                            [-1, self.batch_size, self.n_hidden_dim])
                    lstm_feed_ibp = io.reshape(layer1_ibp,
                            [-1, self.batch_size, self.n_hidden_dim])
                    lstm_feed_ibp = tf.transpose(lstm_feed_ibp, [0, 2, 1, 3])
                
                    def LSTMcond(h__con, c__con, h__ibp, c__ibp, i):
                        return i < tf.shape(lstm_feed_con)[0]

                    def LSTMbody(h__con, c__con, h__ibp, c__ibp, i):
                        h_con, c_con = self.LSTMCell_con(lstm_feed_con[i],
                                h__con, c__con, kernel, bias)
                        h_ibp, c_ibp = self.LSTMCell_ibp(lstm_feed_ibp[i],
                                h__ibp, c__ibp, kernel, bias)
                        return h_con, c_con, h_ibp, c_ibp, i+1

                    lstm_result_con, _, lstm_result_ibp, _, _ = tf.while_loop(
                            LSTMcond, LSTMbody,
                            [tf.zeros([self.batch_size, self.n_hidden_dim]),
                            tf.zeros([self.batch_size, self.n_hidden_dim]),
                            io.zeros([self.batch_size, self.n_hidden_dim]),
                            io.zeros([self.batch_size, self.n_hidden_dim]), 0])
                
                with tf.name_scope('fc2') as layer_scope:
                    self.logits_con = tf.add(tf.matmul(lstm_result_con, W2), b2,
                            name='logits')
                    self.logits_ibp = io.affine(lstm_result_ibp, W2, b2)

                with tf.name_scope('train') as layer_scope:
                    label_onehot = tf.one_hot(self.labels, self.n_digits)
                    label_onehot_neg = tf.subtract(1., label_onehot)
                    self.worst_ibp = tf.add(
                            tf.multiply(io.ub(self.logits_ibp), label_onehot_neg),
                            tf.multiply(io.lb(self.logits_ibp), label_onehot))
                    loss_fit = tf.reduce_mean(tf.losses.softmax_cross_entropy(
                            onehot_labels=tf.one_hot(self.labels, self.n_digits),
                            logits=self.logits_con))
                    loss_spec = tf.reduce_mean(tf.losses.softmax_cross_entropy(
                            onehot_labels=tf.one_hot(self.labels, self.n_digits),
                            logits=self.worst_ibp))
                    self.loss = tf.add(tf.multiply(self.kappa, loss_fit),
                            tf.multiply(tf.subtract(1.,  self.kappa), loss_spec))
                    self.train = tf.train.AdamOptimizer(
                            learning_rate=self.lr).minimize(self.loss)
                
        return graph


    def make_batch(self, inputs, labels):
        maxlen_signal = max(inputs, key=lambda x: x.shape[0])
        maxlen = maxlen_signal.shape[0] + FLAGS.frame_step
        
        batch_inputs = np.zeros([0, maxlen])
        batch_labels = np.zeros([0])

        for inp, lab in zip(inputs, labels):
            padded_input = np.expand_dims(np.concatenate(
                    [inp, np.random.rand(maxlen-inp.shape[0])], 0), 0)
            batch_inputs = np.concatenate([batch_inputs, padded_input], 0)
            batch_labels = np.concatenate([batch_labels, [lab]], 0)
        
        return batch_inputs, batch_labels
    
    
    def kappa_fn(self, e):
        if e < FLAGS.train_epochs / 2.:
            return 1 - e / FLAGS.train_epochs
        else:
            return 0.5
        return 1 - e / FLAGS.train_epochs / 2.
    
    def eps_fn(self, e):
        return -69 - (200 - 69) ** (1 - e / FLAGS.train_epochs)


    def fit(self, dl, verbose=False, resume=False, use_ibp=True,
            accuracy_hold=0.8):
        train_inputs = dl.train_inputs
        train_labels = dl.train_labels
        validation_inputs = dl.validation_inputs
        validation_labels = dl.validation_labels
        test_inputs = dl.test_inputs
        test_labels = dl.test_labels
        
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=3)
            pathlib.Path('./models/'+FLAGS.model_name) \
                    .mkdir(parents=True, exist_ok=True)
            '''
            # Not sure if this is necessary so commented out atm
            if resume:
                try:
                    ckpt = tf.train.latest_checkpoint(self.model_dir)
                    saver.restore(sess, ckpt)
                except ValueError:
                    print('No checkpoint found')
            '''
            stable_epoch = 0
            for epochs in range(FLAGS.train_epochs):
                if verbose:
                    print(f'<Current training epoch = {epochs}>')
                    print(f'\tkappa: {self.kappa_fn(stable_epoch)}')
                    print(f'\teps: {self.eps_fn(stable_epoch)}')

                idx = list(np.random.permutation(len(train_inputs)))
                train_inputs_ = [x for _, x in sorted(zip(idx, train_inputs))]
                train_labels_ = [x for _, x in sorted(zip(idx, train_labels))]

                epoch_mean_loss = 0.
                for i in tqdm(range(0, len(train_inputs), FLAGS.batch_size)):
                    batch_inputs, batch_labels = self.make_batch(
                            train_inputs_[i:i+FLAGS.batch_size],
                            train_labels_[i:i+FLAGS.batch_size])
                    feed_dict = {self.batch_size: batch_inputs.shape[0],
                            self.eps: self.eps_fn(stable_epoch),
                            self.kappa: self.kappa_fn(stable_epoch),
                            self.signal_con: batch_inputs,
                            self.labels: batch_labels}
                    loss, _ = sess.run([self.loss, self.train], feed_dict)
                    epoch_mean_loss += loss
                
                epoch_mean_loss /= np.ceil(len(train_inputs) / FLAGS.batch_size)
                
                if verbose:
                    print(f'\tTraining loss = {epoch_mean_loss}')

                if (epochs + 1) % 1 == 0:
                    if verbose:
                        print('<Performing validation>')

                    vali_mean_loss = 0.
                    correct_count = 0.
                    worst_correct = 0.
                    for i in tqdm(range(0, len(validation_inputs), FLAGS.batch_size)):
                        batch_inputs, batch_labels = self.make_batch(
                                validation_inputs[i:i+FLAGS.batch_size],
                                validation_labels[i:i+FLAGS.batch_size])
                        feed_dict = {self.batch_size: batch_inputs.shape[0],
                                self.eps: self.eps_fn(stable_epoch),
                                self.kappa: self.kappa_fn(stable_epoch),
                                self.signal_con: batch_inputs,
                                self.labels: batch_labels}
                        loss, logits, worst = sess.run(
                                [self.loss, self.logits_con, self.worst_ibp], feed_dict)
                        vali_mean_loss += loss

                        predicted = np.argmax(logits, axis=1)
                        worst_pred = np.argmax(worst, axis=1)
                        correct_count += np.sum(predicted == batch_labels)
                        worst_correct += np.sum(worst_pred == batch_labels)
                    
                    vali_mean_loss /= np.ceil(len(validation_inputs) / FLAGS.batch_size)
                    
                    if verbose:
                        print(f'\tValidation loss = {vali_mean_loss}')
                        print(f'\tAccuracy = {correct_count/len(validation_inputs)*100}%')
                        print(f'\tProvability = {worst_correct/correct_count*100}%')

                    if use_ibp and correct_count / len(validation_inputs) > accuracy_hold:
                        stable_epoch += 1

                saver.save(sess, self.model_dir+'/ckpt', global_step=epochs)
                if verbose:
                    print(f'Saved model at {self.model_dir}')

    def test(self, dl, test_idx=-1, use_ibp=True,
            test_eps=-100.):
        test_inputs = dl.test_inputs
        test_labels = dl.test_labels

        with tf.Session(graph=self.graph) as sess:
            saver = tf.train.Saver()
            try:
                ckpt = tf.train.latest_checkpoint(self.model_dir)
                saver.restore(sess, ckpt)
            except ValueError:
                print('No checkpoint found')

            def single_prediction(input, label):
                single_input = np.expand_dims(input, 0)
                feed_dict = {self.batch_size: 1,
                        self.eps: test_eps,
                        self.kappa: 1.,
                        self.signal_con: single_input,
                        self.labels: [label]}
                        
                logits, worst = sess.run([self.logits_con, self.worst_ibp],
                        feed_dict)
                
                predicted = np.argmax(logits, axis=1)
                worst_pred = np.argmax(worst, axis=1)
                return logits, np.sum(predicted == [label]), \
                        worst, np.sum(worst_pred == [label])
                
            if test_idx < 0:
                print('Test all data')
                correct_count = 0.
                worst_correct = 0.
                for i in tqdm(range(len(test_inputs))):
                    _, res, _, ibpres = single_prediction(test_inputs[i],
                            test_labels[i])
                    correct_count += res
                    worst_correct += ibpres

                print(f'Accuracy = {correct_count/len(test_inputs)*100}%')
                if use_ibp:
                    print(f'Provability = {worst_correct/correct_count*100}%')

            else:
                logits, res, worst, ibpres= single_prediction(
                        test_inputs[test_idx], test_labels[test_idx])
                predicted = np.argmax(logits, axis=1)[0]
                print(f'label: {test_labels[test_idx]}, predicted: {predicted}')
                if res == 1:
                    print('CORRECT')
                else:
                    print('INCORRECT')
                burger = np.concatenate([np.expand_dims(range(self.n_digits), 0),
                        logits], 0)
                print(tabulate(burger, tablefmt='fancygrid'))

                if use_ibp:
                    if ibpres == 1:
                        print('PROVABLE')
                    else:
                        print('ADVERSARIAL MAY EXISTS')
                    burger = np.concatenate([np.expand_dims(range(self.n_digits), 0),
                            worst], 0)
                    print(tabulate(burger, tablefmt='fancygrid'))

