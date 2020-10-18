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

Create batches of these encoded strings. Use the padded_batch method to zero-pad the sequences to the length of the longest string in the batch:



