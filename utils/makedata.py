from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nltk.tag import pos_tag_sents

import codecs
import operator
import numpy as np

# Utility for making the train and test data.

def extractinfo():
	sentlist = list()
	dictlist = list()
	f = codecs.open('../datasets/data1/raw-tagged-sentences.txt', 'r', 'utf-8')
	for line in f: 
		tmp = line.rstrip().split('\t')
		sentlist.append(tmp[0].rstrip())
		dictlist.append(eval(tmp[-2].rstrip()))
	return sentlist, dictlist


def makedata(postaglist, dictlist):
	data = list()
	for sent, d in zip(postaglist, dictlist):
		attrlist = ['NONE'] * len(sent)
		for k in d:
			try: 
				for i in d[k]: 
					attrlist[i] = k
			except:
				print(sent, d)
		newsent = [(tup[0], tup[1], attrlist[i]) for i, tup in enumerate(sent)]
		data.append(newsent)
	return data


def printdata(data):
	f = codecs.open('../datasets/data1/trainset.txt', 'w', 'utf-8')
	f.write('\n'.join([' '.join(['_'.join([t for t in tup]) for tup in sent]) for sent in data]) + '\n')
	f.close()


def makewordvocab():
	vocab = dict()
	f = codecs.open('../datasets/data1/raw-tagged-sentences.txt', 'r', 'utf-8')
	for line in f: 
		toklist = line.rstrip().split('\t')[0].rstrip().split()
		for tok in toklist:
			if tok in vocab: vocab[tok] += 1
			else: vocab[tok] = 1
	f.close()
	sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
	vocabwords = [tup[0] for tup in sorted_vocab]
	f = codecs.open('../datasets/data1/vocabword_train.txt', 'w', 'utf-8')
	f.write('\n'.join(vocabwords) + '\n')
	f.close()
	return vocabwords


def getemb(vocabwords):
	fe = codecs.open('../datasets/data1/glove.text8.100d.txt', 'r', 'utf-8')	# can be created by running GloVe code on Text 8 Corpus
	embdict = dict()
	newemblst = list()
	for line in fe:
		lst = line.rstrip().split(' ')
		embdict[lst[0].lower()] = [float(x) for x in lst[1:]]
	print('Read Embeddings..')
	for i, v in enumerate(vocabwords):
		if v in embdict: newemblst.append(embdict[v])
		else: newemblst.append(np.zeros(100))
		if i % 1000 == 0: print('Done ', i)
	emb = np.array(newemblst)
	np.save('../datasets/data1/wordemb-text8-100', emb)
	return emb
		

def readvocab():
	vocabwords = list()
	f = codecs.open('../datasets/data1/vocabword.txt', 'r', 'utf-8')
	for w in f: vocabwords.append(w.rstrip())
	f.close()
	return vocabwords


def combinevocabs():
	vocab = list()
	f = codecs.open('../datasets/data1/model/vocabword_train.txt', 'r', 'utf-8')
	for w in f: vocab.append(w.rstrip())
	f.close()
	f = codecs.open('../datasets/data1/model/vocabword_test.txt', 'r', 'utf-8')
	for w in f: 
		w = w.rstrip()
		if w not in vocab: vocab.append(w)
	f.close()
	f = codecs.open('../datasets/data1/model/vocabword.txt', 'w', 'utf-8')
	for w in vocab: f.write(w + '\n')
	f.close()
	return vocab


def main():
	sentlist, dictlist = extractinfo()
	print('Info Extracted..')
	postaglist = pos_tag_sents([sent.split() for sent in sentlist], tagset='universal')
	print('POS Tagging Done')
	data = makedata(postaglist, dictlist)
	printdata(data)

	vocabwords = makewordvocab()
	
	# vocabwords = combinevocabs()
	# vocabwords = readvocab()
	# emb = getemb(vocabwords)


if __name__ == '__main__':
	main()
