#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:22:40 2017

@author: dear
"""
import tensorflow as  tf 

def load_data():
    reader_handler = open(image_pair_path, 'r')

    image_one_path_list = []
    image_two_path_list = []
    label_list = []

    count = 0
    for line in reader_handler:
        count = count + 1
        elems = line.split("\t")
        if len(elems) < 3:
            print("len(elems) < 3:" + line)
            continue
        image_one_path = elems[0].strip()
        image_two_path = elems[1].strip()
        label = int(elems[2].strip())

        image_one_path_list.append(image_one_path)
        image_two_path_list.append(image_two_path)
        label_list.append(label)

    return image_one_path_list, image_two_path_list, label_list


# 根据图片路径读取图片
def get_image(image_path):  
    """Reads the jpg image from image_path. 
    Returns the image as a tf.float32 tensor 
    Args: 
        image_path: tf.string tensor 
    Reuturn: 
        the decoded jpeg image casted to float32 
    """  
    content = tf.read_file(image_path)
    tf_image = tf.image.decode_jpeg(content, channels=3)

    return tf_image

def slice_input_producer_demo(image_pair_path, summary_path):
    # 重置graph
    tf.reset_default_graph() 
    # 获取<图片一系统路径，图片二系统路径，标签信息>三个list（load_data函数见supplementary）
    image_one_path_list, image_two_path_list, label_list = load_data()
    ## 构造数据queue
    train_input_queue = tf.train.slice_input_producer([image_one_path_list, image_two_path_list, label_list], capacity=10 * batch_size)

    ## queue输出数据
    img_one_queue = get_image(train_input_queue[0])
    img_two_queue = get_image(train_input_queue[1])
    label_queue = train_input_queue[2]

    ## shuffle_batch批量从queu批量读取数据
    batch_img_one, batch_img_two, batch_label = tf.train.shuffle_batch([img_one_queue, img_two_queue, label_queue],batch_size=batch_size,capacity =  10 + 10* batch_size,min_after_dequeue = 10,num_threads=16,shapes=[(image_width, image_height, image_channel),(image_width, image_height, image_channel),()])

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    summary_writer = tf.train.SummaryWriter(summary_path, graph_def=sess.graph)

    ## 启动queue线程
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)  

    for i in range(10):
        batch_img_one_val, batch_img_two_val, label = sess.run([batch_img_one, batch_img_two,batch_label])
        for k in range(batch_size):
            fig = plt.figure()
            fig.add_subplot(1,2,1)
            plt.imshow(batch_img_one_val[k])
            fig.add_subplot(1,2,2)
            plt.imshow(batch_img_two_val[k])
            plt.show()


    coord.request_stop()  
    coord.join(threads)  
    sess.close()
    summary_writer.close()