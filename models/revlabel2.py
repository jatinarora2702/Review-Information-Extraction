from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import codecs
import os

np.set_printoptions(threshold=np.nan)

flags = tf.flags
flags.DEFINE_string('trainfile', '../datasets/data1/trainset.txt', 'Directory containing training data')
flags.DEFINE_string('testfile', '../datasets/data1/testset.txt', 'Directory containing test data')
flags.DEFINE_string('vocabword_path', '../datasets/data1/vocabword.txt', 'Word vocabulary file')
flags.DEFINE_string('vocabpos_path', '../datasets/data1/vocabpos.txt', 'POS vocabulary file')
flags.DEFINE_string('vocabattr_path', '../datasets/data1/vocabattr.txt', 'Attribute vocabulary file')
flags.DEFINE_string('wemb_path', '../datasets/data1/wordemb-text8-300.npy', 'Embeddings file')
flags.DEFINE_string('batchsize', 500, 'Mini-batch Size')
flags.DEFINE_string('numsteps', 30, 'Number of steps of recurrence (size of context window ie. how many words from the history to depend on)')
flags.DEFINE_string('numlayers', 2, 'Number of LSTM Layers')
flags.DEFINE_string('pembdim', 12, 'POS embeddings dimension')
flags.DEFINE_string('wembdim', 300, 'Word embeddings dimension')
flags.DEFINE_string('aembdim', 5, 'Attribute embedding dimension')
flags.DEFINE_string('decay', 0.5, '')
flags.DEFINE_string('learning_rate', 1.0, '')
flags.DEFINE_string('max_norm', 5, 'Max Norm of gradient after which it will be clipped')
flags.DEFINE_string('max_max_epoch', 10, 'No. of times the LSTM cell state and value are reinitialized and trained on a stream of input')
flags = flags.FLAGS


def readfile(filename):
	data = list()
	f = codecs.open(filename, 'r', 'utf-8')
	for sent in f: data.append([s.split('_') for s in sent.rstrip().split(' ')])
	f.close()
	return data


def readvocab(vocabfile, increment=False):
	vocab = dict()
	f = codecs.open(vocabfile, 'r', 'utf-8')
	for i, w in enumerate(f): 
		if increment: vocab[w.rstrip()] = i + 1
		else: vocab[w.rstrip()] = i
	f.close()
	return vocab


def getindex(word, vocab):
	if word in vocab: return vocab[word]
	return 0


def indexdata(data):
	vocabword = readvocab(flags.vocabword_path, True)
	vocabpos = readvocab(flags.vocabpos_path)
	vocabattr = readvocab(flags.vocabattr_path)
	idata = list()
	for sent in data: 
		newsent = [np.array([getindex(s[0], vocabword), getindex(s[1], vocabpos), getindex(s[2], vocabattr)]) for s in sent]
		newsent = [np.array([0, 0, 0])] * (30 - len(sent)) + newsent
		idata.append(newsent)
	idata = np.array(idata)
	return idata


def readdata():
	traindata = readfile(flags.trainfile)
	testdata = readfile(flags.testfile)
	itraindata = indexdata(traindata)
	itestdata = indexdata(testdata)
	return itraindata, itestdata


def data_iterator(raw_data, batch_size):
	np.random.shuffle(raw_data)
	num_batches = raw_data.shape[0] // batch_size
	if num_batches == 0: raise ValueError("num_batches == 0, decrease batch_size")
	for i in range(num_batches):
		x = raw_data[i*batch_size : (i+1)*batch_size]
		yield x


def makemodel(is_training):
	X = tf.placeholder(tf.int32, [flags.batchsize, flags.numsteps, 3], name='input')
	wemb = np.vstack((np.zeros((1, flags.wembdim)), np.load(flags.wemb_path)))
	pemb = np.identity(flags.pembdim)
	aemb = np.identity(flags.aembdim)

	Ew = tf.constant(wemb, dtype=tf.float32)
	Ep = tf.constant(pemb, dtype=tf.float32)
	Ea = tf.constant(aemb, dtype=tf.float32)

	e_w = tf.nn.embedding_lookup(Ew, X[:, :, 0])
	e_p = tf.nn.embedding_lookup(Ep, X[:, :, 1])
	e = tf.concat([e_w, e_p], 2)
	target = tf.nn.embedding_lookup(Ea, X[:, :, 2])

	lstmdim = flags.wembdim + flags.pembdim
	lstm_cell = tf.contrib.rnn.LSTMCell(lstmdim, state_is_tuple=True)
	lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.9)
	cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * flags.numlayers)
	state = cell.zero_state(flags.batchsize, tf.float32)
	initial_state = state

	lstm_output, final_state = tf.nn.dynamic_rnn(cell, e, initial_state=initial_state)
	outputs = tf.transpose(lstm_output, [1, 0, 2])

	model_outputs = list()
	W = tf.get_variable('weights', shape=[lstmdim, flags.aembdim], dtype=tf.float32)
	b = tf.get_variable('biases', shape=[flags.aembdim], dtype=tf.float32)
	for i in range(flags.numsteps): model_outputs.append(tf.matmul(outputs[i], W) + b)
	
	model_outputs = tf.stack(model_outputs)
	model_outputs = tf.transpose(model_outputs, [1, 0, 2])
	flat_model_outputs = tf.reshape(model_outputs, [-1, flags.aembdim])
	target = tf.reshape(target, [-1, flags.aembdim])
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=flat_model_outputs))
	lr = tf.Variable(0.0, trainable=False)
	if is_training:
		train_vars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(loss, train_vars), 5)
		opt = tf.train.GradientDescentOptimizer(learning_rate = lr)
		optimizer = opt.apply_gradients(zip(grads, train_vars), global_step=tf.contrib.framework.get_or_create_global_step())
	else:
		optimizer = tf.no_op()
	
	modeldict = {'X': X, 'e_w': e_w, 'e_p': e_p, 'target': target, 'initial_state': initial_state, 'final_state': final_state, 'loss': loss, 'optimizer': optimizer, 'lr': lr, 'outputs': model_outputs}
	return modeldict


def get_labels(x, probx):
	labeledx = np.reshape(np.argmax(probx, 2), (x.shape[0], x.shape[1], 1))
	return np.concatenate((x, labeledx), axis=2)


def myfileprint(s):
	if not os.path.exists('../results/model2/data1/'): os.makedirs('../results/model2/data1/')
	f = codecs.open('../results/model2/data1/training-log.txt', 'a', 'utf-8')
	f.write(s + '\n')
	f.close()


def runepoch(sess, data, modeldict, fetches, epoch_no, verbose, record):
	lr_decay = flags.decay ** max(epoch_no - 4, 0.0)
	print('lr_decay: ', lr_decay); myfileprint('lr_decay: ' + str(lr_decay)); 
	sess.run(tf.assign(modeldict['lr'], flags.learning_rate * lr_decay))
	state = sess.run(modeldict['initial_state'])
	losses = 0.0

	if verbose: print('Running New Epoch'); myfileprint('Running New Epoch')
	for curr, x in enumerate(data_iterator(data, flags.batchsize)):
		feed_dict = {modeldict['X']: x, modeldict['initial_state']: state}
		vals = sess.run(fetches, feed_dict)
		losses += vals['loss']
		state = vals['final_state']
		if record: 
			newx = get_labels(x, vals['outputs'])
			f = codecs.open('../results/model2/data1/record-10.txt', 'a', 'utf-8')
			newlist = newx.tolist()
			for sent in newlist:
				for tok in sent: f.write(' '.join([str(t) for t in tok]) + '\n')
				f.write('\n')
			f.close()
		if curr % 10 == 0 and verbose: print('Curr: ', curr, ' | Loss: ', losses); myfileprint('Curr: ' + str(curr) + ' | Loss: ' + str(losses))

	if verbose: print('Epoch Complete'); myfileprint('Epoch Complete')
	return losses


def train(traindata):
	with tf.Session() as sess:
		m_train = makemodel(True)
		print('Model Creation Done')

		fetches = dict()
		fetches['final_state'] = m_train['final_state']
		fetches['loss'] = m_train['loss']
		fetches['optimizer'] = m_train['optimizer']

		# fetches['e_w'] = m_train['e_w']
		# fetches['e_p'] = m_train['e_p']
		# fetches['target'] = m_train['target']

		print('Training Started')
		saver = tf.train.Saver(max_to_keep=50)
		tf.train.write_graph(sess.graph_def, '../modelparams/model2/data1/', 'model.pb', as_text=False)
		sess.run(tf.global_variables_initializer())
		# saver.restore(sess, '../modelparams/model2-data1/checkpoint-9')
		for i in range(flags.max_max_epoch):
			runepoch(sess, traindata, m_train, fetches, i, True, False)
			save_path = saver.save(sess, '../modelparams/model2/data1/model')
			save_path = saver.save(sess, '../modelparams/model2/data1/checkpoint-' + str(i))
			print("Model saved in file: %s" % save_path); myfileprint("Model saved in file: " + str(save_path))
		save_path = saver.save(sess, '../modelparams/model2/data1/model')
		print('Training Complete')


def test(testdata):
	with tf.Session() as sess:
		m_test = makemodel(False)
		fetches = dict()
		print('Testing'); myfileprint('Testing')
		fetches['final_state'] = m_test['final_state']
		fetches['loss'] = m_test['loss']		
		fetches['outputs'] = m_test['outputs']
		saver = tf.train.Saver(max_to_keep=50)
		saver.restore(sess, '../modelparams/model2/data1/model')
		p = runepoch(sess, testdata, m_test, fetches, 0, True, True)
		print(p); myfileprint(str(p))


def main():
	traindata, testdata = readdata()
	print('Traindata Shape: ', traindata.shape)
	print('Testdata Shape: ', testdata.shape)
	# train(traindata)
	test(testdata)


if __name__ == '__main__':
	main()
