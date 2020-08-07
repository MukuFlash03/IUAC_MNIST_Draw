import numpy as np
from tensorflow import keras

#Model Definition#
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

def model_compile(_optimizer, _loss, _metrics):
 model.compile(optimizer=_optimizer,loss=_loss,metrics=[_metrics])

def model_training(_epochs):
 model.fit(train_images, train_labels, epochs=_epochs)
 test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
 results=[test_loss, test_acc]
 return results

def predict(_image):
 _image=np.array(_image)
 _image=_image.reshape(1,28,28)
 predictions = model.predict(_image)
 preds = []
 for i in range(0,10):
     preds.append(predictions[0][i])
 return preds