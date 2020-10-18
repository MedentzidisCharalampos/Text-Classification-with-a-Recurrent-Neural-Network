# Text-Classification-with-a-Recurrent-Neural-Network
This text classification model trains a recurrent neural network on the IMDB large movie review dataset for sentiment analysis.

# The dataset
The IMDB large movie review dataset is a binary classification dataset—all the reviews have either a positive or negative sentiment.

Download the dataset using TFDS(https://www.tensorflow.org/datasets).


# Prepare the data for training
We use tfds.deprecated.text.SubwordTextEncoder with 8k vocab size. This text encoder will reversibly encode any string.  
An Example:  
The original string: "Hello TensorFlow."  
Encoded string is [4025, 222, 6307, 2327, 4043, 2120, 7975]  
4025 ----> Hell  
222 ----> o   
6307 ----> Ten  
2327 ----> sor  
4043 ----> Fl  
2120 ----> ow  
7975 ----> .  

Create batches of these encoded strings. Use the padded_batch method to zero-pad the sequences to the length of the longest string in the batch.

# Create the model
Build a tf.keras.Sequential model and start with an embedding layer.
1. An embedding layer stores one vector per word. When called, it converts the sequences of word indices to sequences of vectors. These vectors are trainable. After training (on enough data), words with similar meanings often have similar vectors.  
Note: This index-lookup is much more efficient than the equivalent operation of passing a one-hot encoded vector through a tf.keras.layers.Dense layer.

2. A recurrent neural network (RNN) processes sequence input by iterating through the elements. RNNs pass the outputs from one timestep to their input—and then to the next.
The tf.keras.layers.Bidirectional wrapper can also be used with an RNN layer. This propagates the input forward and backwards through the RNN layer and then concatenates the output. This helps the RNN to learn long range dependencies.





