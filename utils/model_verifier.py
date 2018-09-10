from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import numpy as np

# Utility to verifying/printing the word embeddings for words in the data vocabulary. 

def main():
	emb = np.load('data/wordemb.npy')
	f = codecs.open('data/vocabword.txt')
	for i, w in enumerate(f): 
		print(w)
		print(emb[i])
		print('======================================')
		if i > 10: break
	f.close()


if __name__ == '__main__':
	main()
