# -*- encoding:utf8 -*-
import tensorflow as tf
import numpy as np
import os
import sys
import logging as logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger_name="qacnn"
stdout = sys.stdout
reload(sys)
sys.stdout = stdout

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Get ENV
ENVIRON = os.environ.copy()
import cPickle as pkl
from utils import *
from models import QACNN


class QACNNConfig(object):
    def __init__(self, vocab_size, embeddings=None):
        # max question length
        self.max_q_length = 200
        # max answer length
        self.max_a_length = 200
        # epochs
        self.num_epochs = 100
        # batch size
        self.batch_size = 128
        # vocabulary size
        self.vocab_size = vocab_size
        # worde embedding size(dimession)
        self.embeddings = embeddings
        self.embedding_size = 100
        if self.embeddings is not None:
            self.embedding_size = embeddings.shape[1]
        # filter size: like 1-gram or 3,5 gram
        self.filter_sizes = [1, 2, 3, 5, 7, 9]
        # hidden layer size
        self.hidden_size = 128
        # filter 
        self.num_filters = 128
        self.l2_reg_lambda = 0.
        self.keep_prob = 0.5
        # lr
        self.lr = 0.001
        # margin
        self.m = 0.5

        self.cf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.cf.gpu_options.per_process_gpu_memory_fraction = 0.2
        

def train(train_corpus, config, val_corpus, eval_train_corpus=None):
    iterator = Iterator(train_corpus)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    with tf.Session(config=config.cf) as sess:
        model = QACNN(config)
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        for epoch in xrange(config.num_epochs):
            count = 0
            for batch_x in iterator.next(config.batch_size, shuffle=True):
                batch_q, batch_ap, batch_an = zip(*batch_x)
                batch_q = np.asarray(batch_q)
                batch_ap = np.asarray(batch_ap)
                batch_an = np.asarray(batch_an)
                _, loss, accu = sess.run([model.train_op, model.total_loss, model.accu], 
                                   feed_dict={model.q:batch_q, 
                                              model.aplus:batch_ap, 
                                              model.aminus:batch_an,
                                              model.keep_prob:config.keep_prob})
                count += 1
                if count % 10 == 0:
                    print('[epoch {}, batch {}]Loss:{}, Accuracy:{}'.format(epoch, count, loss, accu))
            saver.save(sess,'{}/my_model'.format(model_path), global_step=epoch)
            if eval_train_corpus is not None:
                train_res = evaluate(sess, model, eval_train_corpus, config)
                print('[train] ' + train_res)
            if val_corpus is not None:
                val_res = evaluate(sess, model, val_corpus, config)
                print('[eval] ' + val_res)


def evaluate(sess, model, corpus, config):
    iterator = Iterator(corpus)

    count = 0
    total_qids = []
    total_aids = []
    total_pred = []
    total_labels = []
    total_loss = 0.
    tryone= 0
    for batch_x in iterator.next(config.batch_size, shuffle=False):
        if(tryone == 0):
            batch_qids, batch_q, batch_aids, batch_ap, labels = zip(*batch_x)
            #print(batch_qids)
            #print(batch_q)
            print(type(batch_q))
            #print(batch_aids)
            #print(batch_ap)
            #print(labels)
            tryone+=1
        batch_qids, batch_q, batch_aids, batch_ap, labels = zip(*batch_x)
        batch_q = np.asarray(batch_q)
        batch_ap = np.asarray(batch_ap)
        q_ap_cosine, loss = sess.run([model.q_ap_cosine, model.loss], 
                           feed_dict={model.q:batch_q, 
                                      model.aplus:batch_ap, 
                                      model.aminus:batch_ap,
                                      model.keep_prob:1.})
        total_loss += loss
        count += 1
        total_qids.append(batch_qids)
        total_aids.append(batch_aids)
        total_pred.append(q_ap_cosine)
        total_labels.append(labels)
        #print(batch_qids[0], [id2word[_] for _ in batch_q[0]], 
         #    batch_aids[0], [id2word[_] for _ in batch_ap[0]])
    total_qids = np.concatenate(total_qids, axis=0)
    total_aids = np.concatenate(total_aids, axis=0)
    total_pred = np.concatenate(total_pred, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    MAP, MRR = eval_map_mrr(total_qids, total_aids, total_pred, total_labels)
    return 'MAP:{}, MRR:{}'.format(MAP, MRR)
                
def predictShort(sess, model, sentence, config,in_file,corpus):
    iterator = Iterator(corpus)
    q=buildQ(sentence,word2id)
    listq=[]
    for x in range(config.batch_size):
        listq.append(q)
    list_q=tuple(listq)
    for batch_x in iterator.next(config.batch_size, shuffle=False):
        batch_qids, batch_q, batch_aids, batch_ap, labels = zip(*batch_x)
        batch_q = np.asarray(list_q)
        batch_ap = np.asarray(batch_ap)
        q_ap_cosine, loss = sess.run([model.q_ap_cosine, model.loss], 
                           feed_dict={model.q:batch_q, 
                                      model.aplus:batch_ap, 
                                      model.aminus:batch_ap,
                                      model.keep_prob:1.})
    
    #print(q_ap_cosine)
    total_qids = np.concatenate(total_qids, axis=0)
    total_aids = np.concatenate(total_aids, axis=0)
    total_pred = np.concatenate(total_pred, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    

def test(corpus, config):
    with tf.Session(config=config.cf) as sess:
        model = QACNN(config)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        print('[test] ' + evaluate(sess, model, corpus, config))
                    
def predict(sentence,config,in_file,corpus):
    with tf.Session(config=config.cf) as sess:
        model = QACNN(config)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        print('[predict] ' + predictShort(sess, model, sentence, config,in_file,corpus))
def main(args):
    max_q_length = 25
    max_a_length = 90
    with open(os.path.join(processed_data_path, 'pairwise_corpus.pkl'), 'r') as fr:
        train_corpus, val_corpus, test_corpus = pkl.load(fr)
    with open(os.path.join(processed_data_path, 'pointwise_corpus.pkl'), 'r') as fr:
        eval_train_corpus, _, _ = pkl.load(fr)
   
    
    embeddings = build_embedding(embedding_path, word2id)
    train_q, train_ap, train_an = zip(*train_corpus)
    train_q = padding(train_q, max_q_length)
    print("train_q")
    print(train_q[0])
    train_ap = padding(train_ap, max_a_length)
    print(train_ap[0])
    train_an = padding(train_an, max_a_length)
    train_corpus = zip(train_q, train_ap, train_an)
    val_qids, val_q, val_aids, val_ap, labels = zip(*val_corpus)
    val_q = padding(val_q, max_q_length)
    val_ap = padding(val_ap, max_a_length)
    val_corpus = zip(val_qids, val_q, val_aids, val_ap, labels)

    eval_train_qids, eval_train_q, eval_train_aids, eval_train_ap, eval_train_labels = zip(*eval_train_corpus)
    eval_train_q = padding(eval_train_q, max_q_length)
    eval_train_ap = padding(eval_train_ap, max_a_length)
    eval_train_corpus = zip(eval_train_qids, eval_train_q, eval_train_aids, eval_train_ap, eval_train_labels)

    test_qids, test_q, test_aids, test_ap, labels = zip(*test_corpus)
    test_q = padding(test_q, max_q_length)
    test_ap = padding(test_ap, max_a_length)
    test_corpus = zip(test_qids, test_q, test_aids, test_ap, labels)

    config = QACNNConfig(max(word2id.values()) + 1, embeddings=embeddings)
    config.max_q_length = max_q_length
    config.max_a_length = max_a_length
    if args.train:
        train(train_corpus, config, val_corpus, eval_train_corpus)
    elif args.test:
        test(test_corpus, config)
    elif args.predict:
        print(args.predict)
        predict(args.predict, config,embedding_path,train_corpus)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train",  help="whether to train", action='store_true')
    parser.add_argument("--test",  help="whether to test", action='store_true')
    parser.add_argument("--predict", "-i", default="what city is oregon state university in" , help="predict sentence",type=str)
    args = parser.parse_args()
    raw_data_path = '../data/WikiQA/raw'
    processed_data_path = '../data/WikiQA/processed'
    processed_data_pkl = os.path.join(processed_data_path, 'vocab.pkl')
    embedding_path = '../data/embedding/glove.6B.100d.txt'
    model_path = 'models'

    if 'GLOVE_EMBEDDING_6B' in ENVIRON:
        embedding_path = ENVIRON['GLOVE_EMBEDDING_6B']


    if not os.path.exists(processed_data_pkl): 
        raise BaseException("data [%] not exist, run ch6/preprocess_wiki.py first." % processed_data_pkl)

    with open(os.path.join(processed_data_path, 'vocab.pkl'), 'r') as fr:
        word2id, id2word = pkl.load(fr)
    main(args)
