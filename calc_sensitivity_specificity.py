def calc(THRESHOLD, y_pred, y_true):
	tp = 0
	p = 0
	tn = 0
	n = 0
	for i, val in enumerate([t >= THRESHOLD for t in y_pred]):
		if val: p += 1
		if val and y_true[i] >= .9999: tp += 1
		if not val: n += 1
		if not val and y_true[i] <= .0001: tn += 1
	return (tp, p, tn, n)	# sensitivity: tp/p and specificity: tn/n

