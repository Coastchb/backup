#By @Kevin Xu
#kevin28520@gmail.com
#Youtube: https://www.youtube.com/channel/UCVCSn4qQXTDAtGWpWAe4Plw
#
#The aim of this project is to use TensorFlow to process our own data.
#    - input_data.py:  read in data and generate batches
#    - model: build the model architecture
#    - training: train

# I used Ubuntu with Python 3.5, TensorFlow 1.0*, other OS should also be good.
# With current settings, 10000 traing steps needed 50 minutes on my laptop.

# data: cats vs. dogs from Kaggle
# Download link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
# data size: ~540M

# How to run?
# 1. run the training.py once
# 2. call the run_training() in the console to train the model.

# Note: 
# it is suggested to restart your kenel to train the model multiple times 
#(in order to clear all the variables in the memory)
# Otherwise errors may occur: conv1/weights/biases already exist......


#%%

import os
#import numpy as np
import tensorflow as tf
import input_data
import model

#%%
train_feat_dir = "/nfs/user/caohaibing/spyder/dear/data/train/feat" #'/home/dear/spyder/dear/data/train/feat'
train_lab_dir = "/nfs/user/caohaibing/spyder/dear/data/train/phn_lab" #'/home/dear/spyder/dear/data/train/phn_lab'
train_log_dir = "nfs/user/caohaibing/spyder/dear/log" #'/home/dear/spyder/dear/log'
N_CLASSES = 20
BATCH_SIZE = 128
#CAPACITY = 2000
MAX_EPOCH = 100 #10000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001
momentum = 0.9
shuffle_each_epoch = True

with_header = True
start_index = 5;
  
#%%
def run_training(train_feat_dir, train_lab_dir, train_log_dir, with_header, start_index):
      
    train_file_paths, train_labels, feat_dim, max_time_step = input_data.get_files(train_feat_dir, train_lab_dir, with_header, start_index)        
    
    train_file_batchs, train_label_batchs = input_data.get_batchs(train_file_paths,
                                                          train_labels,
                                                          feat_dim,
                                                          BATCH_SIZE)
    
  
    print("generating graph...");

    inputX = tf.placeholder(tf.float32, shape=(max_time_step, BATCH_SIZE, feat_dim))
    targetIxs = tf.placeholder(tf.int64)     
    targetVals = tf.placeholder(tf.int32)			
    targetShape = tf.placeholder(tf.int64)
    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)			
    seqLengths = tf.placeholder(tf.int32, shape=(BATCH_SIZE))
    
    logits, logits3d = model.inference(BATCH_SIZE, feat_dim, inputX, max_time_step, N_CLASSES)

    train_loss, train_op = model.optimization(logits3d, seqLengths, targetY, learning_rate, momentum)
    
    error_rate = model.evaluation(logits, logits3d, seqLengths, targetY)
       
#    summary_op = tf.summary.merge_all()
    
    sess = tf.Session()
#    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    for epoch in range(MAX_EPOCH):
        print('epoch %d...' % epoch)
        
        batch_error_rates = [];
        for batch_index in range(len(train_file_batchs)):
            step = epoch * len(train_file_batchs) + batch_index;
            
            file_batch, label_batch = train_file_batchs[batch_index], train_label_batchs[batch_index];

            batch_inputs, batch_target_sparse, batch_seq_lens = input_data.read_batch_data(file_batch, label_batch, max_time_step, with_header, start_index);

            batch_target_ixs,  batch_target_vals, batch_target_shape = batch_target_sparse
            feed_dict = {inputX: batch_inputs, targetIxs: batch_target_ixs, targetVals: batch_target_vals,
					targetShape: batch_target_shape, seqLengths: batch_seq_lens}

            tra_loss, _, tra_err_rate = sess.run([train_loss, train_op, error_rate], feed_dict=feed_dict)
               
            if step % 10 == 0:
                print('--Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, (1 - tra_err_rate)*100.0))
               # summary_str = sess.run(summary_op)
               # train_writer.add_summary(summary_str, step)
            
            if step % 10 == 0 or batch_index == (len(train_file_batchs) - 1):
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
            batch_error_rates.append(tra_err_rate * BATCH_SIZE);
            
#            print('--step %d: error rate: %.3f' % (step, tra_err_rate));
		
        epoch_error_rate = sum(batch_error_rates) / (BATCH_SIZE * len(train_file_batchs));
        
        print('epoch mean error rate: %.3f\n' % epoch_error_rate)
        
        if(shuffle_each_epoch and (epoch < (MAX_EPOCH - 1))):
            train_file_batchs, train_label_batchs = input_data.get_batchs(train_file_paths,
                                                          train_labels,
                                                          feat_dim,
                                                          max_time_step,
                                                          BATCH_SIZE)

    sess.close()
    
#%%
#train model
run_training(train_feat_dir, train_lab_dir, train_log_dir, with_header, start_index);

#%% Evaluate one image
# when training, comment the following codes.


#from PIL import Image
#import matplotlib.pyplot as plt
#
#def get_one_image(train):
#    '''Randomly pick one image from training data
#    Return: ndarray
#    '''
#    n = len(train)
#    ind = np.random.randint(0, n)
#    img_dir = train[ind]
#
#    image = Image.open(img_dir)
#    plt.imshow(image)
#    image = image.resize([208, 208])
#    image = np.array(image)
#    return image
#
#def evaluate_one_image():
#    '''Test one image against the saved models and parameters
#    '''
#    
#    # you need to change the directories to yours.
#    train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
#    train, train_label = input_data.get_files(train_dir)
#    image_array = get_one_image(train)
#    
#    with tf.Graph().as_default():
#        BATCH_SIZE = 1
#        N_CLASSES = 2
#        
#        image = tf.cast(image_array, tf.float32)
#        image = tf.image.per_image_standardization(image)
#        image = tf.reshape(image, [1, 208, 208, 3])
#        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
#        
#        logit = tf.nn.softmax(logit)
#        
#        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
#        
#        # you need to change the directories to yours.
#        logs_train_dir = '/home/kevin/tensorflow/cats_vs_dogs/logs/train/' 
#                       
#        saver = tf.train.Saver()
#        
#        with tf.Session() as sess:
#            
#            print("Reading checkpoints...")
#            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#            if ckpt and ckpt.model_checkpoint_path:
#                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                saver.restore(sess, ckpt.model_checkpoint_path)
#                print('Loading success, global_step is %s' % global_step)
#            else:
#                print('No checkpoint file found')
#            
#            prediction = sess.run(logit, feed_dict={x: image_array})
#            max_index = np.argmax(prediction)
#            if max_index==0:
#                print('This is a cat with possibility %.6f' %prediction[:, 0])
#            else:
#                print('This is a dog with possibility %.6f' %prediction[:, 1])


#%%





