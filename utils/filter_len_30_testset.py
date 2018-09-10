from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs

# Utility to filter and remove all sentence from dataset, of length greater than 30.

def main():
	data = list()
	f = codecs.open('data/testset_orig.txt', 'r', 'utf-8')
	for sent in f:
		sentlist = sent.rstrip().split(' ')
		if len(sentlist) <= 30: data.append(sent)
	f.close()
	f = codecs.open('data/testset.txt', 'w', 'utf-8')
	for sent in data: f.write(sent)
	f.write('\n')
	f.close()


if __name__ == '__main__':
	main()
