# -*- encoding:utf-8 -*-
import tensorflow as tf
import numpy as np

class QACNN(object):
    """
    pairwise model
    """
    def __init__(self, config):
        self.config = config
        # 输入
        self.add_placeholders()
        # [batch_size, sequence_size, embed_size]
        q_embed, aplus_embed, aminus_embed = self.add_embeddings()
        # [batch_size, sequence_size, hidden_size, 1]
        self.h_q, self.h_ap, self.h_am = self.add_hl(q_embed, aplus_embed, aminus_embed)
        # [batch_size, total_channels]
        real_pool_q, real_pool_ap, real_pool_am = self.add_model(q_embed, aplus_embed, aminus_embed)
        # [batch_size, 1]
        self.q_ap_cosine, self.q_am_cosine = self.calc_cosine(real_pool_q, real_pool_ap, real_pool_am)
        # loss and accu
        self.total_loss, self.loss, self.accu = self.add_loss_op(self.q_ap_cosine, self.q_am_cosine)
        # train op
        self.train_op = self.add_train_op(self.total_loss)
        # word sum
        self.sum_f = self.add_sum(self.sum_a,self.sum_b)


    # tf placeholder
    def add_placeholders(self):
        # question
        self.q = tf.placeholder(tf.int32,
                shape=[None, self.config.max_q_length],
                name='Question')
        # pos answer
        self.aplus = tf.placeholder(tf.int32,
                shape=[None, self.config.max_a_length],
                name='PosAns')
        # neg answer
        self.aminus = tf.placeholder(tf.int32,
                shape=[None, self.config.max_a_length],
                name='NegAns')
        # drop_out
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.batch_size = tf.shape(self.q)[0]
        #sum
        self.sum_a=tf.placeholder(tf.int32)
        self.sum_b=tf.placeholder(tf.int32)
    #word sum
    def add_sum(self,sum_a,sum_b):
        total=0
        print("call add sum")
        
        return sum_a+sum_b
    # word embeddings
    def add_embeddings(self):
        with tf.variable_scope('embedding'):
            if self.config.embeddings is not None:
                embeddings = tf.Variable(self.config.embeddings, name="embeddings", trainable=False)
            else:
                embeddings = tf.get_variable('embeddings', shape=[self.config.vocab_size, self.config.embedding_size], initializer=tf.uniform_unit_scaling_initializer())
            q_embed = tf.nn.embedding_lookup(embeddings, self.q)
            aplus_embed = tf.nn.embedding_lookup(embeddings, self.aplus)
            aminus_embed = tf.nn.embedding_lookup(embeddings, self.aminus)
            q_embed = tf.nn.dropout(q_embed, keep_prob=self.keep_prob)
            aplus_embed = tf.nn.dropout(aplus_embed, keep_prob=self.keep_prob)
            aminus_embed = tf.nn.dropout(aminus_embed, keep_prob=self.keep_prob)
            return q_embed, aplus_embed, aminus_embed

    # Hidden Layer
    def add_hl(self, q_embed, aplus_embed, aminus_embed):
        with tf.variable_scope('HL'):
            W = tf.get_variable('weights', shape=[self.config.embedding_size, self.config.hidden_size], initializer=tf.uniform_unit_scaling_initializer())
            b = tf.get_variable('biases', initializer=tf.constant(0.1, shape=[self.config.hidden_size]))
            h_q = tf.reshape(tf.nn.tanh(tf.matmul(tf.reshape(q_embed, [-1, self.config.embedding_size]), W)+b), [-1, self.config.max_q_length, self.config.hidden_size])
            h_ap = tf.reshape(tf.nn.tanh(tf.matmul(tf.reshape(aplus_embed, [-1, self.config.embedding_size]), W)+b), [-1, self.config.max_a_length, self.config.hidden_size])
            h_am = tf.reshape(tf.nn.tanh(tf.matmul(tf.reshape(aminus_embed, [-1, self.config.embedding_size]), W)+b), [-1, self.config.max_a_length, self.config.hidden_size])
            tf.add_to_collection('total_loss', 0.5*self.config.l2_reg_lambda*tf.nn.l2_loss(W))
            return h_q, h_ap, h_am

    # CNN layer
    def add_model(self, h_q, h_ap, h_am):
        pool_q = list()
        pool_ap = list()
        pool_am = list()
        h_q = tf.reshape(h_q, [-1, self.config.max_q_length, self.config.embedding_size, 1])
        h_ap = tf.reshape(h_ap, [-1, self.config.max_a_length, self.config.embedding_size, 1])
        h_am = tf.reshape(h_am, [-1, self.config.max_a_length, self.config.embedding_size, 1])
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.variable_scope('filter{}'.format(filter_size)):
                conv1_W = tf.get_variable('W_q', shape=[filter_size, self.config.embedding_size, 1, self.config.num_filters], initializer=tf.truncated_normal_initializer(.0, .1))
                conv2_W = tf.get_variable('W_a', shape=[filter_size, self.config.embedding_size, 1, self.config.num_filters], initializer=tf.truncated_normal_initializer(.0, .1))
                conv1_b = tf.get_variable('conv_qb', initializer=tf.constant(0.1, shape=[self.config.num_filters]))
                conv2_b = tf.get_variable('conv_ab', initializer=tf.constant(0.1, shape=[self.config.num_filters]))
                # pooling layer bias,Q and A split
                pool_qb = tf.get_variable('pool_qb', initializer=tf.constant(0.1, shape=[self.config.num_filters]))
                pool_ab = tf.get_variable('pool_ab', initializer=tf.constant(0.1, shape=[self.config.num_filters]))
                # cnn
                out_q = tf.nn.relu((tf.nn.conv2d(h_q, conv1_W, [1,1,1,1], padding='VALID')+conv1_b))
                # pool
                out_q = tf.nn.max_pool(out_q, [1,self.config.max_q_length-filter_size+1,1,1], [1,1,1,1], padding='VALID')
                out_q = tf.nn.tanh(out_q+pool_qb)
                pool_q.append(out_q)

                out_ap = tf.nn.relu((tf.nn.conv2d(h_ap, conv2_W, [1,1,1,1], padding='VALID')+conv2_b))
                out_ap = tf.nn.max_pool(out_ap, [1,self.config.max_a_length-filter_size+1,1,1], [1,1,1,1], padding='VALID')
                out_ap = tf.nn.tanh(out_ap+pool_ab)
                pool_ap.append(out_ap)

                out_am = tf.nn.relu((tf.nn.conv2d(h_am, conv2_W, [1,1,1,1], padding='VALID')+conv2_b))
                out_am = tf.nn.max_pool(out_am, [1,self.config.max_a_length-filter_size+1,1,1], [1,1,1,1], padding='VALID')
                out_am = tf.nn.tanh(out_am+pool_ab)
                pool_am.append(out_am)

                # regulation
                tf.add_to_collection('total_loss', 0.5*self.config.l2_reg_lambda*tf.nn.l2_loss(conv1_W))
                tf.add_to_collection('total_loss', 0.5*self.config.l2_reg_lambda*tf.nn.l2_loss(conv2_W))

        total_channels = len(self.config.filter_sizes)*self.config.num_filters
        real_pool_q = tf.reshape(tf.concat(pool_q, 3), [-1, total_channels])
        real_pool_ap = tf.reshape(tf.concat(pool_ap, 3), [-1, total_channels])
        real_pool_am = tf.reshape(tf.concat(pool_am, 3), [-1, total_channels])

        return real_pool_q, real_pool_ap, real_pool_am

    # cosine
    def calc_cosine(self, real_pool_q, real_pool_ap, real_pool_am):
        normalized_q_h_pool = tf.nn.l2_normalize(real_pool_q, dim=1)
        normalized_pos_h_pool = tf.nn.l2_normalize(real_pool_ap, dim=1)
        normalized_neg_h_pool = tf.nn.l2_normalize(real_pool_am, dim=1)
        q_ap_cosine = tf.reduce_sum(tf.multiply(normalized_q_h_pool, normalized_pos_h_pool),1)
        q_am_cosine = tf.reduce_sum(tf.multiply(normalized_q_h_pool, normalized_neg_h_pool),1)

        return q_ap_cosine, q_am_cosine

    # loss op default set m=0.5
    def add_loss_op(self, q_ap_cosine, q_am_cosine):
        original_loss = self.config.m - q_ap_cosine + q_am_cosine
        l = tf.maximum(tf.zeros_like(original_loss), original_loss)
        loss = tf.reduce_sum(l)
        tf.add_to_collection('total_loss', loss)
        total_loss = tf.add_n(tf.get_collection('total_loss'))
        accu = tf.reduce_mean(tf.cast(tf.equal(0., l), tf.float32))
        return total_loss, loss, accu

    # train op
    def add_train_op(self, loss):
        with tf.name_scope('train_op'):
            # train step
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            opt = tf.train.AdamOptimizer(self.config.lr)
            train_op = opt.minimize(loss, self.global_step)
            return train_op