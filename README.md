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
Note: The tf.keras.layers.Bidirectional wrapper can also be used with an RNN layer. This propagates the input forward and backwards through the RNN layer and then concatenates the output. This helps the RNN to learn long range dependencies.

3. The fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units.

4. The last layer is densely connected with a single output node.

Compile and train the model using the Adam optimizer and BinaryCrossentropy loss.
# Results

| Accuracy and Validation Accuracy      | Loss and Validation Loss      |
|------------|-------------|

| <img src="https://github.com/MedentzidisCharalampos/Text-Classification-with-a-Recurrent-Neural-Network/blob/main/accuracy1lstm.png" > |
<img src="https://github.com/MedentzidisCharalampos/Text-Classification-with-a-Recurrent-Neural-Network/blob/main/Loss_epochs_first_model.png"> |

# Stack Two LSTM Layers
We examine another architecture of the model.
Keras recurrent layers have two available modes that are controlled by the return_sequences constructor argument:

1. Return either the full sequences of successive outputs for each timestep (a 3D tensor of shape (batch_size, timesteps, output_features)).
2. Return only the last output for each input sequence (a 2D tensor of shape (batch_size, output_features)).

This architecture follows the first model in order to stack two  LSTM Layers.  
