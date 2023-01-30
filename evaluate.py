import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

# load model
model = tf.keras.models.load_model('saved_model/my_model')

# load data
# kika på https://www.tensorflow.org/tutorials/load_data/images
(train_images, train_labels), (test_images, test_labels) = \
    keras.datasets.mnist.load_data()

# # check its architecture
# model.summary()

# # evaluate
# test_loss, test_acc = model.evaluate(test_images, test_labels)

# load webcam
webcam = cv2.VideoCapture(0)

while True:
    # capture frame
    ret, frame = webcam.read()
    
    if ret:
        # convert to black and white
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (thresh, bw_frame) = cv2.threshold(gray_frame, 90, 255, cv2.THRESH_BINARY)
        
        # display feed
        cv2.imshow("ScannerX2000", bw_frame)
        key = cv2.waitKey(1)
        if key == 13: # enter
            break

# resize to 28x28
dim = (28, 28)
small_frame = cv2.resize(bw_frame, dim, interpolation=cv2.INTER_CUBIC)        

# convert to numpy array
np_image_data = np.asarray(small_frame)
np_final = np.expand_dims(np_image_data, axis=0)

# make prediction
prediction = model.predict(np_final)
print(f"Prediction: {np.argmax(prediction, axis=-1)}")