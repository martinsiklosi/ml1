import tensorflow as tf
from tensorflow import keras

# get data
(train_images, train_labels), (test_images, test_labels) = \
    keras.datasets.mnist.load_data()

# setup model
model = keras.sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(120, activasion=tf.nn.relu)
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train model
model.fit(train_images, train_labels, epochs=5)

# evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test accuracy {test_acc}")

# make predictions
predictions = model.predict(test_images)