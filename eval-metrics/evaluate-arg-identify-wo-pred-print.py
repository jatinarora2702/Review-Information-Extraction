from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs

# Utility for measuring the overall F1-Score, Precision, Recall and Accuracy for Argument (except Predicte) Identification task, at a given model checkpoint

def evaluate(num):
	f = codecs.open('../results/model3-data1/record-' + str(num) + '.txt', 'r', 'utf-8')
	data = list()
	sent = list()
	for line in f:
		line = line.rstrip()
		if line == '': data.append(sent); sent = list()
		else: lst = sent.append([int(t) for t in line.split()])
	f.close()

	tp = tn = fp = fn = 0
	for sent in data:
		for d in sent:
			if d[2] > 0 and d[2] < 4: 
				if d[3] > 0 and d[3] < 4: tp += 1
				else: fn += 1
			else: 
				if d[3] > 0 and d[3] < 4: fp += 1
				else: tn += 1
	p = tp / max(tp + fp, 1.0)
	r = tp / max(tp + fn, 1.0)
	f1 = (2.0 * p * r) / max(p + r, 1.0)
	a = (tp + tn) / max(tp + tn + fp + fn, 1.0)
	
	f = codecs.open('../results/model3-data1/results-arg-identify-wo-pred-all.txt', 'a', 'utf-8')
	f.write(str(num) + ', ' + str(p) + ', ' + str(r) + ', ' + str(f1) + ', ' + str(a) + '\n')	

	# f.write('precision: ' + str(p) + '\n')
	# f.write('recall: ' + str(r) + '\n')
	# f.write('f-score: ' + str(f1) + '\n')
	# f.write('accuracy: ' + str(a) + '\n')

	f.close()


def main():
	for i in range(10, 11): evaluate(i)


if __name__ == '__main__':
	main()
