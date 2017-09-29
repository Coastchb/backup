
#By @Kevin Xu
#kevin28520@gmail.com
#Youtube: https://www.youtube.com/channel/UCVCSn4qQXTDAtGWpWAe4Plw
#

#The aim of this project is to use TensorFlow to process our own data.
#    - input_data.py:  read in data and generate batches
#    - model: build the model architecture
#    - training: train

# data: cats vs. dogs from Kaggle
# Download link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
# data size: ~540M

# How to run?
# 1. run the training.py once
# 2. call the run_training() in the console to train the model.


#%%

import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
import numpy as np
#%%
def inference(batch_size, feat_dim, inputX, max_time_step, n_classes):
    '''Build the model
    
    ''' 
    #Prep input data to fit requirements of rnn.bidirectional_rnn
    #  Reshape to 2-D tensor (nTimeSteps*batchSize, feat_dim)			
    inputXrs = tf.reshape(inputX, [-1, feat_dim])
    
    #  Split to get a list of 'n_steps' tensors of shape (batch_size, feat_dim);similar to inputX		
    inputList = tf.split(inputXrs, max_time_step, 0)			
   
    nHidden = 128

		
    ####Weights & biases			
    weightsOutH1 = tf.Variable(tf.truncated_normal([2, nHidden],
                                                   stddev=np.sqrt(2.0 / (2*nHidden))))			
    biasesOutH1 = tf.Variable(tf.zeros([nHidden]))			
#    weightsOutH2 = tf.Variable(tf.truncated_normal([2, nHidden],
#                                                   stddev=np.sqrt(2.0 / (2*nHidden))))			
#    biasesOutH2 = tf.Variable(tf.zeros([nHidden]))			
    weightsClasses = tf.Variable(tf.truncated_normal([nHidden, n_classes],
                                                     stddev=np.sqrt(2.0 / nHidden)))			
    biasesClasses = tf.Variable(tf.zeros([n_classes]))

			
    ####Network			
    forwardH1 = tf.contrib.rnn.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)			
    backwardH1 = tf.contrib.rnn.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)	
		
    fbH1, _, _ = tf.contrib.rnn.static_bidirectional_rnn(forwardH1, backwardH1, inputList, dtype=tf.float32)		
    fbH1rs = [tf.reshape(t, [batch_size, 2, nHidden]) for t in fbH1]			
    outH1 = [tf.reduce_sum(tf.multiply(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs]
		
    logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]
				
    logits3d = tf.stack(logits)

    return logits, logits3d     			


#%%
def optimization(logits3d, seqLengths, targetY, learningRate, momentum):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]
        
    Returns:
        loss tensor of float type
    '''					
    loss = tf.reduce_mean(ctc.ctc_loss(targetY, logits3d, seqLengths))			
    optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(loss)

    return loss, optimizer;


#%%
def evaluation(logits3d, seqLengths, targetY):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqLengths)[0][0])   
      
  errorRate = tf.reduce_sum(tf.edit_distance(predictions, targetY, normalize=False)) / \
			tf.to_float(tf.size(targetY.values)) 

  return errorRate
