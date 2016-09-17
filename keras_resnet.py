from __future__ import print_function
import random, glob, sys

import keras
from keras.layers import merge
from sklearn.utils.class_weight import compute_class_weight
from convnets import *
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import load_img, img_to_array
import keras.backend as K
from keras.metrics import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, History, LearningRateScheduler
from keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T

class LogHistory(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		f = open("plots_"+NAME+"/{}.history".format(epoch+194), "w")
		f.write(str(logs))
		f.close()
		
NAME = "resnet_224"

img_rows = img_cols = IMG_ROWS = IMG_COLS = 224
train_data_dir = "train"
test_data_dir = "test"
validation_data_dir = "validation"
batch_size = 16
WEIGHTS_FN = "weights_"+NAME+"/{epoch:03d}-{val_loss:.2f}.hdf5"
RNG_SEED = 123454321R
DROPOUT = .4
NB_EPOCH = 100000 # rely on early stopping!
LR = 0.0003333
LR_SCHEDULE = {}
W_REG = 1.0
ALPHA = .02


def scheduler(epoch):
	global LR
	if epoch in LR_SCHEDULE:
		LR = LR_SCHEDULE[epoch]
	return LR

total_train_0 = len(glob.glob(train_data_dir+"/0/*.jpeg", recursive=True))
total_train = len(glob.glob(train_data_dir+"/**/*.jpeg", recursive=True))
total_val = len(glob.glob(validation_data_dir+"/**/*.jpeg", recursive=True))
total_test = len(glob.glob(test_data_dir+"/**/*.jpeg", recursive=True))

sys.setrecursionlimit(10000)
random.seed(RNG_SEED)
np.random.seed(random.randint(0, 4294967295))

def clip(inp):
	return K.clip(inp, 0, 1)
	
def normalize(img):
	img -= K.mean(img)
	img /= K.std(img)
	return img
def ninety(mat):
	T.set_subtensor(mat[0], K.transpose(mat[0][::-1]))
	T.set_subtensor(mat[1], K.transpose(mat[1][::-1]))
	T.set_subtensor(mat[2], K.transpose(mat[2][::-1]))
	return mat
def one_eighty(mat):
	return mat[:, ::-1]
def neg_ninety(mat):
	T.set_subtensor(mat[0], K.transpose(mat[0])[::-1])
	T.set_subtensor(mat[1], K.transpose(mat[1])[::-1])
	T.set_subtensor(mat[2], K.transpose(mat[2])[::-1])
	return mat

def f1_score(y_true, y_pred):
	y_pred = K.round(y_pred)
	tp = K.sum(K.equal(y_pred, 1) * K.equal(y_true, 1))
	fp = K.sum(K.equal(y_pred, 1) * K.equal(y_true, 0))
	fn = K.sum(K.equal(y_pred, 0) * K.equal(y_true, 1))
	precision = K.switch(K.equal(tp, 0), 0, tp/(tp+fp))
	recall = K.switch(K.equal(tp, 0), 0, tp/(tp+fn))
	return K.switch(K.equal(precision+recall, 0), 0, 2*(precision*recall)/(precision+recall))

if __name__ == '__main__':
	internal = ResNet_50()
	for layer in internal.layers:
		layer.W_regularizer = l2(W_REG)
	inp2 = Input((3, IMG_ROWS, IMG_COLS))
	model = Lambda(normalize)(inp2)
	b = Lambda(ninety)(model)
	c = Lambda(one_eighty)(model)
	d = Lambda(neg_ninety)(model)
	a = internal(model)
	b = internal(b)
	c = internal(c)
	d = internal(d)
	if DROPOUT is not None:
		a = Dropout(DROPOUT)(a)
		b = Dropout(DROPOUT)(b)
		c = Dropout(DROPOUT)(c)
		d = Dropout(DROPOUT)(d)
	model = merge([a,b,c,d], mode="ave")
	model = Dense(1024, W_regularizer=l2(W_REG))(model)
	model = LeakyReLU(ALPHA)(model)
	model = Dense(512, W_regularizer=l2(W_REG))(model)
	model = LeakyReLU(ALPHA)(model)
	model = Dense(1, W_regularizer=l2(W_REG))(model)
	model = Activation("sigmoid")(model)
	#model = LeakyReLU(ALPHA)(model)
	#model = Lambda(clip, output_shape=lambda x:x)(model)
	model = Model(inp2, model)
	model.summary()
	for layer in model.layers:
		layer.W_regularizer = l2(W_REG)
	model.compile(loss="binary_crossentropy",
		#	optimizer=SGD(lr=LR, momentum=.9, nesterov=True),
			optimizer=Adam(lr=LR),
			metrics=["accuracy", f1_score])
	train_datagen = ImageDataGenerator(
			horizontal_flip=True,
			vertical_flip=True)
			
	test_datagen = ImageDataGenerator()

	train_generator = train_datagen.flow_from_directory(
			train_data_dir,
			target_size=(img_rows, img_cols),
			batch_size=batch_size,
			class_mode="binary",
			shuffle=True,
		   )
			
	validation_generator = test_datagen.flow_from_directory(
			validation_data_dir,
			target_size=(img_rows, img_cols),
			batch_size=batch_size,
			class_mode="binary",
		   )

	test_generator = test_datagen.flow_from_directory(
			test_data_dir,
			target_size=(img_rows, img_cols),
			batch_size=batch_size,
			class_mode="binary",
		 )
	total_train_1 = total_train-total_train_0
	cw = compute_class_weight("balanced", np.array([0,1]), np.array([0]*total_train_0 + [1]*total_train_1))
	cw = {0: cw[0], 1:cw[1]}
	print(cw)
	try:
		history = History()
		model.fit_generator(
			train_generator,
			samples_per_epoch=total_train,
			nb_epoch=NB_EPOCH,
			validation_data=validation_generator,
			nb_val_samples=total_val,
			callbacks=[
				ModelCheckpoint(WEIGHTS_FN),
				history,
				LogHistory(),
				LearningRateScheduler(scheduler)
				],
			class_weight=cw
			)
	except KeyboardInterrupt:
		pass
	print(history.history)
	f = open("plots_"+NAME+"/history.txt", "w")
	f.write(str(history.history))
	f.close()
	model.save_weights(WEIGHTS_FN)
	try:
		val_loss, = plt.plot(history.history["val_loss"], "r-", label="Validation Loss")
		loss, = plt.plot(history.history["loss"], "b-", label="Training Loss")
		plt.title("Loss scores (scaled binary crossentropy)")
		plt.legend(handles=[val_loss, loss])
		plt.savefig("plots_"+NAME+"/loss_fig.png")
		plt.close()
		val_f1, = plt.plot(history.history["val_f1_score"], "r-", label="Validation F1")
		f1, = plt.plot(history.history["f1_score"], "b-", label="Test F1")
		plt.title("F1 scores")
		plt.legend(handles=[val_f1, f1])
		plt.savefig("plots_"+NAME+"/f1_fig.png")
		plt.close()
	except:
		pass
	y_pred = []
	y_true = []
	for x, y in test_generator:
		y_pred += model.predict_on_batch(x).tolist()
		y_true += y.tolist()
		if len(y_pred) >= total_test:
			break
	f = open("plots_"+NAME+"/final_eval_pred.txt", "w")
	f.write(str(y_pred))
	f.close()
	f = open("plots_"+NAME+"/final_eval_true.txt", "w")
	f.write(str(y_true))
	f.close()
	scores = model.evaluate_generator(test_generator, val_samples=total_test)
	print(scores)
