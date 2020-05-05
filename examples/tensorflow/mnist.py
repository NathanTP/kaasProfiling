import tensorflow as tf
import os

# Set the number of epochs to train
epochs = 2

mnist = tf.keras.datasets.mnist
# path must be absolute because keras interprets it relative to its cache directory
(x_train, y_train), (x_test, y_test) = mnist.load_data(path=os.path.join(os.getcwd(), 'data/mnist.npz'))
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

print("Compiling model")
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

print("training model")
model.fit(x_train, y_train, epochs=epochs)

accuracy = model.evaluate(x_test,  y_test, verbose=2)[1]
print("Accuracy:", accuracy)
