from tensorflow import keras
import cv2
import numpy as np

# load model
model = keras.models.load_model('saved_model/my_model')
model.load_weights('saved_model/weights.h5')

# # load data
# # kika på https://www.tensorflow.org/tutorials/load_data/images
# (train_images, train_labels), (test_images, test_labels) = \
#     keras.datasets.mnist.load_data()


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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #(thresh, frame) = cv2.threshold(frame, 10, 255, cv2.THRESH_BINARY)
        
        # increase contrast
        contrast = 100
        brightness = 2
        frame = cv2.convertScaleAbs(frame, contrast, brightness)

        # # posterize
        # frame[frame > 120] = 255
        # frame[frame < 100] = 0

        # # dilate
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        # frame = cv2.erode(frame, kernel)

        # make square
        height = min(np.shape(frame))
        frame = frame[:height,:height]

        # display feed
        cv2.imshow("ScannerX2000", frame)
        key = cv2.waitKey(1)
        if key == 13: # enter
            cv2.destroyAllWindows()
            break
      
# resize to 28x28
dim = (28, 28)
picture_data = cv2.resize(frame, dim, interpolation=cv2.INTER_CUBIC)     

# convert to numpy array
picture_data = np.asarray(picture_data)
picture_data = np.expand_dims(picture_data, axis=0)
picture_data = np.invert(picture_data)

# make prediction
prediction = model.predict(picture_data)
prediction = str(np.argmax(prediction, axis=-1)[0])
print(f"{prediction=}")

# show taken image
cv2.imshow(f"{prediction}", frame)
key = cv2.waitKey(10000)
if key == 13: # enter
    cv2.destroyAllWindows()
