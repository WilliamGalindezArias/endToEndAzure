from keras.models import load_model
from keras import backend
import numpy as np


def load_model_custom():
	return load_model('trained_model/model_cnn_1.h5')


def predict(img):
	test=(255-img.reshape(1,28,28,1).astype('float32'))/255
    # Predict from Keras
	one_hot = load_model_custom().predict(test)
	backend.clear_session()
	return np.argmax(one_hot)

