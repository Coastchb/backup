
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 12:25:00 2017

@author: dear
"""

# -*- coding:utf-8 -*-
import tensorflow as tf

file_list = ["csv/file01.csv", "csv/file02.csv"];
label_list = [0, 1];

input_queue = tf.train.slice_input_producer([file_list, label_list])

labels = input_queue[1];

feats = tf.read_file(input_queue[0])
print(feats)
# Default values, in case of empty columns. Also specifies the type of the
# decoded result.

#record_defaults = [[1], [1], [1]]
#col1, col2, col3 = tf.decode_csv(feats, record_defaults = record_defaults)
#features = tf.stack([col1, col2])

init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()  # local variables like epoch_num, batch_size 可以不初始化local
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(local_init_op)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(5):
        # Retrieve a single instance:
        #example, label = sess.run([features, col3])
        labels = sess.run(labels)
#        print(example)
        print(labels)

    coord.request_stop()
    coord.join(threads)