from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nltk.tag import map_tag
from random import shuffle
import codecs
import operator

# Utility for converting data from CONLL format, to that required by the model.

def get_attribute(tok):
	if '\tY\t' in tok: return 'PRED'
	elif '\tA0' in tok: return 'ASP'
	elif '\tA1' in tok: return 'PROD1'
	elif '\tA2' in tok: return 'PROD2'
	else: return 'NONE'


def make_data():
	data_conll = list()
	tags = list()
	f = codecs.open('../datasets/data1/testdata_conll.txt', 'r', 'utf-8')
	for s in f:
		s = s.rstrip()
		if s == '': data_conll.append(tags); tags = list()
		else: tags.append(s)
	f.close()

	data = list()
	for sent in data_conll:
		newsent = list()
		for tok in sent:
			toklist = tok.split('\t')
			pos = map_tag('en-ptb', 'universal', toklist[5])
			if pos != '.':
				word = toklist[3]
				attr = get_attribute(tok)
				newsent.append([word, pos, attr])
		data.append(newsent)
	return data


def make_vocab():
	f = codecs.open('../datasets/data1/model/test.txt', 'r', 'utf-8')
	vocab = dict()
	for sent in f:
		toklist = [tok.split('_')[0] for tok in sent.split(' ')]
		for tok in toklist:
			if tok in vocab: vocab[tok] += 1
			else: vocab[tok] = 1
	f.close()
	sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
	f = codecs.open('../datasets/data1/model/vocabword_test.txt', 'w', 'utf-8')
	f.write('\n'.join([tup[0] for tup in sorted_vocab]))
	f.write('\n')
	f.close()


def make_testdata():
	dataset = make_data()
	shuffle(dataset)
	f = codecs.open('../datasets/data1/testset.txt', 'w', 'utf-8')
	for sent in dataset:
		s = ' '.join('_'.join(tok) for tok in sent)
		f.write(s + '\n')
	f.close()


def main():
	make_testdata()
	# make_vocab()


if __name__ == '__main__':
	main()
