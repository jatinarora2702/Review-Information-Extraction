from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs

# Utility for measuring the the F1-Score, Precision, Recall and Accuracy for Argument Classification task, at a given model checkpoint

def evaluate(num):
	f = codecs.open('../results/model3-data1/record-' + str(num) + '.txt', 'r', 'utf-8')
	data = list()
	sent = list()
	for line in f:
		line = line.rstrip()
		if line == '': data.append(sent); sent = list()
		else: lst = sent.append([int(t) for t in line.split()])
	f.close()

	tp = [0, 0, 0, 0, 0]
	tn = [0, 0, 0, 0, 0]
	fp = [0, 0, 0, 0, 0]
	fn = [0, 0, 0, 0, 0]
	
	for sent in data:
		for d in sent:
			if d[3] == d[2]: 
				for x in range(5): tn[x] += 1
				tp[d[2]] += 1
				tn[d[2]] -= 1
			else: 
				for x in range(5): tn[x] += 1
				fp[d[3]] += 1
				tn[d[3]] -= 1
				fn[d[2]] += 1
				tn[d[2]] -= 1
	
	p = [0.0, 0.0, 0.0, 0.0, 0.0]
	r = [0.0, 0.0, 0.0, 0.0, 0.0]
	f1 = [0.0, 0.0, 0.0, 0.0, 0.0]
	a = [0.0, 0.0, 0.0, 0.0, 0.0]
	
	for x in range(5):
		p[x] = tp[x] / max(tp[x] + fp[x], 1.0)
		r[x] = tp[x] / max(tp[x] + fn[x], 1.0)
		if p[x] > 0.0 and r[x] > 0.0: f1[x] = 2.0 / ((1.0 / p[x]) + (1.0 / r[x]))
		else: f1[x] = 0.0
		a[x] = (tp[x] + tn[x]) / max(tp[x] + tn[x] + fp[x] + fn[x], 1.0)
	
	f = codecs.open('../results/model3-data1/results-all.txt', 'a', 'utf-8')
	
	# f.write('NONE | PROD1 | PROD2 | ASP | PRED\n')
	# f.write('precision: ' + str(p) + '\n')
	# f.write('recall: ' + str(r) + '\n')
	# f.write('f-score: ' + str(f1) + '\n')
	# f.write('accuracy: ' + str(a) + '\n')
	
	results = list()
	for k in range(1, 5): results.append([str(p[k]), str(r[k]), str(f1[k]), str(a[k])])
	f.write(str(num) + ', ')
	for elem in results: f.write(', '.join(elem) + ', ')
	f.write('\n')
	f.close()


def main():
	for i in range(10, 11): evaluate(i)


if __name__ == '__main__':
	main()
