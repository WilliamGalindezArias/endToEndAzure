from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
from convnetwork import predict as predictor
app = Flask(__name__)


@app.route('/')
def home():
	return render_template('index.html', data={'status': False})


@app.route('/charrecognize', methods=['POST'])
def predict():
	if request.method == 'POST':
		# I send an image encoded in a JSON File
		data = request.get_json()
		# Data holds now the request received in the POST
		imagebase_64 = data['image']
		# I assign imagebase64 to the image coming from the POST
		img_bytes = base64.b64decode(imagebase_64)
		with open("temp.jpg", "wb") as temp:
			temp.write(img_bytes)
		_, img = cv2.threshold(cv2.imread('temp.jpg',0),127,255,0)
		img = cv2.resize(img, (28, 28))
		return jsonify({'prediction': int(predictor(img)), 'status': True})



if __name__ == '__main__':
    app.run()