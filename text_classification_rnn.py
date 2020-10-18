#Setup

#!pip install -q tfds-nightly

import tensorflow_datasets as tfds
import tensorflow as tf

#Import matplotlib and create a helper function to plot graphs:

import matplotlib.pyplot as plt

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()

#Setup input pipeline

#The IMDB large movie review dataset is a binary classification dataset—all the reviews have either a positive or negative sentiment.

#Download the dataset using TFDS (https://www.tensorflow.org/datasets).

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

#The dataset info includes the encoder (a tfds.deprecated.text.SubwordTextEncoder).

encoder = info.features['text'].encoder
print('Vocabulary size: {}'.format(encoder.vocab_size))
#Vocabulary size: 8185

#This text encoder will reversibly encode any string, falling back to byte-encoding if necessary.

sample_string = 'Hello TensorFlow.'

encoded_string = encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print('The original string: "{}"'.format(original_string))

#Encoded string is [4025, 222, 6307, 2327, 4043, 2120, 7975]
#The original string: "Hello TensorFlow."

assert original_string == sample_string
for index in encoded_string:
  print('{} ----> {}'.format(index, encoder.decode([index])))

#4025 ----> Hell
#222 ----> o
#6307 ----> Ten
#2327 ----> sor
#4043 ----> Fl
#2120 ----> ow
#7975 ----> .

#Prepare the data for training
#Next create batches of these encoded strings.
# Use the padded_batch method to zero-pad the sequences to the length of the longest string in the batch:

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE)

test_dataset = test_dataset.padded_batch(BATCH_SIZE)

#Create the model

#Build a tf.keras.Sequential model and start with an embedding layer.
# An embedding layer stores one vector per word.
# When called, it converts the sequences of word indices to sequences of vectors.
# These vectors are trainable.
# After training (on enough data), words with similar meanings often have similar vectors.
#This index-lookup is much more efficient than the equivalent operation of passing a one-hot encoded vector through a tf.keras.layers.Dense layer.

#A recurrent neural network (RNN) processes sequence input by iterating through the elements.
# RNNs pass the outputs from one timestep to their input—and then to the next.
#The tf.keras.layers.Bidirectional wrapper can also be used with an RNN layer.
# This propagates the input forward and backwards through the RNN layer and then concatenates the output.
# This helps the RNN to learn long range dependencies.

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

#Compile the Keras model to configure the training process:

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

#Train the model

history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset,
                    validation_steps=30)

#loss: 0.1126 - accuracy: 0.9626 - val_loss: 0.4225 - val_accuracy: 0.8469
test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

#loss: 0.4171 - accuracy: 0.8455

#The above model does not mask the padding applied to the sequences.
# This can lead to skew if trained on padded sequences and test on un-padded sequences.
# Ideally you would use masking to avoid this, but as you can see below it only have a small effect on the output.
#If the prediction is >= 0.5, it is positive else it is negative.

def pad_to_size(vec, size):
  zeros = [0] * (size - len(vec))
  vec.extend(zeros)
  return vec

def sample_predict(sample_pred_text, pad):
  encoded_sample_pred_text = encoder.encode(sample_pred_text)

  if pad:
    encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
  encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
  predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

  return (predictions)

# predict on a sample text without padding.

sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print(predictions)
#[[0.07037611]]

# predict on a sample text with padding

sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print(predictions)
#[[-0.26897871]]

plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')

#Stack two or more LSTM layers

#Keras recurrent layers have two available modes that are controlled by the return_sequences constructor argument:
#Return either the full sequences of successive outputs for each timestep (a 3D tensor of shape (batch_size, timesteps, output_features)).
#Return only the last output for each input sequence (a 2D tensor of shape (batch_size, output_features)).

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset,
                    validation_steps=30)

#loss: 0.0948 - accuracy: 0.9764 - val_loss: 0.5652 - val_accuracy: 0.8505

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

#loss: 0.5370 - accuracy: 0.8510

# predict on a sample text without padding.

sample_pred_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=False)
print(predictions)

#[[-2.3630884]]

# predict on a sample text with padding

sample_pred_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print(predictions)

#[[-2.690089]]

plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')

