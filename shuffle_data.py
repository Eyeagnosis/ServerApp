import glob
import random
import os


def shuffle_data(num_validation, num_test, mapping=\
{"0":"0", "1":"1", "2":"1", "3":"1", "4":"1"},\
RNG_seed=None, balance=True):
	if RNG_seed is not None:
		random.seed(RNG_seed)
	img_files = glob.glob("train/**/*.jpeg", recursive=True)
	img_files += glob.glob("validation/**/*.jpeg", recursive=True)
	img_files += glob.glob("test/**/*.jpeg", recursive=True)
	img_files += glob.glob("unused/**/*.jpeg", recursive=True)

	img_files.sort(key=lambda x: os.path.basename(x))
	random.shuffle(img_files)

	labels = open("trainLabels.csv").read().strip().split("\n")
	d = {}
	if not os.path.isdir("validation"):
		os.mkdir("validation")
	if not os.path.isdir("train"):
		os.mkdir("train")
	if not os.path.isdir("test"):
		os.mkdir("test")
	for i in labels:
		t = i.split(",")
		try:
			int(t[1])
			d[t[0]] = mapping[t[1]]
			if not os.path.isdir("validation/{}".format(d[t[0]])):
				os.mkdir("validation/{}".format(d[t[0]]))
			if not os.path.isdir("test/{}".format(d[t[0]])):
				os.mkdir("test/{}".format(d[t[0]]))
			if not os.path.isdir("train/{}".format(d[t[0]])):
				os.mkdir("train/{}".format(d[t[0]]))
		except:
			pass
			
	
	if balance:
		a = {}
		for i in img_files[:num_validation]:
			basename = os.path.basename(i)
			fn, _ = os.path.splitext(basename)
			if d[fn] not in a:
				a[d[fn]] = 0
			a[d[fn]] += 1
		min1 = 2e9
		for k in a:
			min1 = min(a[k], min1)
		tot = {}
	
			
	for i in img_files[:num_validation]:
		basename = os.path.basename(i)
		fn, _ = os.path.splitext(basename)
		if balance:
			if d[fn] not in tot:
				tot[d[fn]] = 0
			if tot[d[fn]] >= min1:
				os.rename(i, "unused/{}".format(basename))
				continue
			tot[d[fn]] += 1
		os.rename(i, "validation/{0}/{1}".format(d[fn], basename))

	for i in img_files[num_validation:num_validation+num_test]:
		basename = os.path.basename(i)
		fn, _ = os.path.splitext(basename)
		os.rename(i, "test/{0}/{1}".format(d[fn], basename))
	
	if balance:
		a = {}
		for i in img_files[num_validation+num_test:]:
			basename = os.path.basename(i)
			fn, _ = os.path.splitext(basename)
			if d[fn] not in a:
				a[d[fn]] = 0
			a[d[fn]] += 1
		min1 = 2e9
		for k in a:
			min1 = min(a[k], min1)
		tot = {}
	
	for i in img_files[num_validation+num_test:]:
		basename = os.path.basename(i)
		fn, _ = os.path.splitext(basename)
		if balance:
			if d[fn] not in tot:
				tot[d[fn]] = 0
			if tot[d[fn]] >= min1:
				os.rename(i, "unused/{}".format(basename))
				continue
			tot[d[fn]] += 1
		os.rename(i, "train/{0}/{1}".format(d[fn], basename))
	
	vals = {}
	
	for subdir, q, r in os.walk("train/"):
		if not os.listdir(subdir):
			os.rmdir(subdir)
		g = len(glob.glob(subdir+"/*.jpeg"))
		if g:
			if "train" not in vals:
				vals["train"] = {}
			vals["train"][subdir[len("train/"):]] = g
		
	for subdir, _, _ in os.walk("test/"):
		if not os.listdir(subdir):
			os.rmdir(subdir)
		g = len(glob.glob(subdir+"/*.jpeg"))
		if g:
			if "test" not in vals:
				vals["test"] = {}
			vals["test"][subdir[len("test/"):]] = g
	
	for subdir, _, _ in os.walk("validation/"):
		if not os.listdir(subdir):
			os.rmdir(subdir)
		g = len(glob.glob(subdir+"/*.jpeg"))
		if g:
			if "validation" not in vals:
				vals["validation"] = {}
			vals["validation"][subdir[len("validation/"):]] = g
			
	return vals
if __name__ == "__main__":
	num_validation = int(input("Num validation?: "))
	num_test = int(input("Num test?: "))
	try:
		RNG_seed = int(input("RNG seed (optional): "))
	except:
		RNG_seed = None
	shuffle_data(num_validation, num_test, RNG_seed=RNG_seed, balance=False)

# find . -type f -print | wc -l
