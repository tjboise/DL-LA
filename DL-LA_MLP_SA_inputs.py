import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.activations import relu
from traceloader import TraceConfig
import numpy
import logging
from tensorflow.keras.optimizers import Adam



# Specify the raw data set to be analyzed (e.g.)
traceset = 'unprotected GIFT-COFB'
traceconfig = TraceConfig()
tracelength = traceconfig.getnrpoints(traceset)


# Define the training and validation parameters
nrtrain = 1400
nrval = 580
nrepochs = 80
batchsize = 32
if nrtrain > 100000:
    nrsensi = 100000
else:
    nrsensi = nrtrain
balance = 1


# Call the trace preparation function to create the required training and validation set
train_x, train_y, val_x, val_y = traceconfig.prep_traces(traceset, nrtrain, nrval, balance)


# Define the model, train the model and validate the model using the created data sets
model = Sequential(
    [Dense(300, activation = 'relu', input_shape= (tracelength,) ),
    BatchNormalization(),
    Dense(200, activation = 'relu'),
    BatchNormalization(),
    Dense(16, activation = 'relu'),
    BatchNormalization(),
    Dense(2, activation = 'softmax') ])

optimizer = Adam(learning_rate=0.001)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
out=model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=nrepochs, batch_size=batchsize, verbose=1)


# Dump the validation accuracy into an ASCII file
logging.basicConfig(filename='val_acc.log', format='%(message)s', level=logging.INFO)
logging.info("val_accuracy: {}".format(out.history['val_accuracy']))

import matplotlib.pyplot as plt
plt.plot(out.history['accuracy'], label = 'accuracy')
plt.plot(out.history['loss'],label = 'loss')
plt.plot(out.history['val_accuracy'],label = 'val_accuracy')
plt.legend()
# plt.plot(out.history['tra_accuracy'])
plt.show()

# Perform sensitivity analysis based on the training set
inp = tf.Variable(train_x[:nrsensi], dtype=tf.float32)
with tf.GradientTape() as tape:
    tape.watch(inp)
    preds = model(inp)
    trues = tf.Variable(train_y[:nrsensi], dtype=tf.float32)
    loss = tf.keras.losses.mean_squared_error(trues, preds)
    grads = tape.gradient(loss, inp)

grads_sum = numpy.sum(numpy.abs(grads), axis=0)


# Dump the sensitivity analysis results into a binary file
f = open("sensi.dat","wb")
f.write(grads_sum.astype("double"))
f.close()

# ... [rest of your code remains unchanged]

# Get the predictions from the trained model for train_x
# y_hat = model.predict(train_x)

# print('y_hat',y_hat)
