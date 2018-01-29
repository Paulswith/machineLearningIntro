
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
# import os



# # Basic model parameters as external flags.
# records_path_train = '/Users/v_ljiayili/Dropbox/Main/MachineLearning/DataSet/train.tfrecords'
# records_path_test = '/Users/v_ljiayili/Dropbox/Main/MachineLearning/DataSet/test.tfrecords'
RECORD_TRAIN = '/Users/v_ljiayili/Dropbox/Main/MachineLearning/DataSet/train.tfrecords'
RECORD_TEST = '/Users/v_ljiayili/Dropbox/Main/MachineLearning/DataSet/test.tfrecords'
FILE_PLACE = '/Users/v_ljiayili/Desktop/train.tfrecords'
MODEL_SAVEPATH = 'save/plane/cnn_plane_2_model.ckpt'
MODEL_SAVEPATH_LAST = 'save/plane_last/cnn_plane_2_model.ckpt'
LOG_PATH = 'Logs/plane'





# ========================================= 索取输入 =========================================
IMG_HEIGHT = 300
IMG_WIDTH = 300
IMG_CHANNELS = 3
IMG_PIXELS = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS


def read_and_decode(filename_queue):
    #创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    #从文件中读出一个样例
    _,serialized_example = reader.read(filename_queue)
    #解析读入的一个样例
    features = tf.parse_single_example(serialized_example,features={
        'label_binary':tf.FixedLenFeature([],tf.int64),
        'image_raw':tf.FixedLenFeature([],tf.string)
        })
    #将字符串解析成图像对应的像素数组
    image = tf.decode_raw(features['image_raw'],tf.uint8)
    label = tf.cast(features['label_binary'],tf.int32)

    image.set_shape([IMG_PIXELS])
    image = tf.reshape(image,[IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS])
    image = tf.cast(image, tf.float32) * (1. / 255.0) - 0.5

    return image,label

#用于获取一个batch_size的图像和label
def inputs(record_file, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer([record_file], num_epochs=num_epochs)
    image,label = read_and_decode(filename_queue)
    #随机获得batch_size大小的图像和label
    images,labels = tf.train.shuffle_batch([image, label],
        batch_size=batch_size,
        num_threads=4,
        capacity=1000 + 3 * batch_size,
        min_after_dequeue=1000
    )
    # labels = np.array(labels).reshape(batch_size, 1)
    return images,labels

def log_p(info, detail):
    print('{a} :::: {b}'.format(a=info, b=detail))

# In[3]:
def tran_two_d(array, dont=None):
    if dont:
        return array
    '''
    将一维的补齐为二维
    :param array:
    :return:
    '''
    new = []
    for i in array:
        if i == 1:
            new.append(0)
        else:
            new.append(1)
    new_np = np.array(new)
    tran_t_d = np.c_[array, new_np.T]
    # print('tran_t_d',tran_t_d)
    # print('tran_t_d.shape',tran_t_d.shape)
    return tran_t_d

def keep(_rate, x):
    if _rate == None:
        return x
    return _rate
# ========================================= 参数初始化 =========================================
# 参数About
r_h = 3
r_w = 3
n_output = 2
stddev = 0.01
learning_rate = 0.00001


with tf.name_scope('Weights'):
    Weights = {
        # `[filter_height, filter_width, in_channels, out_channels]`
        'wc_1': tf.Variable(learning_rate * tf.random_normal([r_h, r_w, 3, 64], stddev=stddev), name='wc_1'),
        'wc_2': tf.Variable(learning_rate * tf.random_normal([r_h, r_w, 64, 128], stddev=stddev), name='wc_2'),
        'wc_3': tf.Variable(learning_rate * tf.random_normal([r_h, r_w, 128, 512], stddev=stddev), name='wc_3'),
        # 'wc_4': tf.Variable(tf.random_normal([r_h, r_w, 256, 512], stddev=stddev), name='wc_4'),
        # 'wc_5': tf.Variable(tf.random_normal([r_h, r_w, 512, 1024], stddev=stddev), name='wc_5'),

        # MAXPOOLING 2x2
        # MaxPooling之后是4x4x512 ,[8192,1024] 
        'w_fc_1': tf.Variable(learning_rate * tf.random_normal([38*38*512, 512], stddev=stddev), name='w_fc_1'),
        # 全连接层2, 接上层的output
        'w_fc_2': tf.Variable(learning_rate * tf.random_normal([512, 64], stddev=stddev), name='w_fc_2'),
        # 'w_fc_3': tf.Variable(tf.random_normal([256, 32], stddev=stddev), name='w_fc_3'),
        'w_output': tf.Variable(learning_rate * tf.random_normal([64, n_output], stddev=stddev), name='w_output')
        # 'w_output_': tf.Variable(tf.random_normal([2, n_output], stddev=stddev), name='w_output')
    }

    tf.summary.histogram('wc_1', Weights['wc_1'])
    tf.summary.histogram('wc_2', Weights['wc_2'])
    tf.summary.histogram('wc_3', Weights['wc_3'])
    # tf.summary.histogram('wc_4', Weights['wc_4'])
    # tf.summary.histogram('wc_5', Weights['wc_5'])
    tf.summary.histogram('w_fc_1', Weights['w_fc_1'])
    tf.summary.histogram('w_fc_2', Weights['w_fc_2'])
    # tf.summary.histogram('w_fc_3', Weights['w_fc_3'])
    tf.summary.histogram('w_output', Weights['w_output'])
    
with tf.name_scope('biases'):
    biases = {
        'bc_1': tf.Variable(learning_rate * tf.random_normal([64], stddev=stddev), name='bc_1'),
        'bc_2': tf.Variable(learning_rate * tf.random_normal([128], stddev=stddev), name='bc_2'),
        'bc_3': tf.Variable(learning_rate * tf.random_normal([512], stddev=stddev), name='bc_3'),
        # 'bc_4': tf.Variable(tf.random_normal([512], stddev=stddev), name='bc_4'),
        # 'bc_5': tf.Variable(tf.random_normal([1024], stddev=stddev), name='bc_5'),
        'b_fc_1': tf.Variable(learning_rate * tf.random_normal([512], stddev=stddev), name='b_fc_1'),
        'b_fc_2': tf.Variable(learning_rate * tf.random_normal([64], stddev=stddev), name='b_fc_2'),
        # 'b_fc_3': tf.Variable(tf.random_normal([32], stddev=stddev), name='b_fc_3'),
        'b_output': tf.Variable(learning_rate * tf.random_normal([n_output], stddev=stddev), name='b_output'),
    }
    tf.summary.histogram('bc_1', biases['bc_1'])
    tf.summary.histogram('bc_2', biases['bc_2'])
    tf.summary.histogram('bc_3', biases['bc_3'])
    # tf.summary.histogram('bc_4', biases['bc_4'])
    # tf.summary.histogram('bc_5', biases['bc_5'])
    tf.summary.histogram('b_fc_1', biases['b_fc_1'])
    tf.summary.histogram('b_fc_2', biases['b_fc_2'])
    # tf.summary.histogram('b_fc_3', biases['b_fc_3'])
    tf.summary.histogram('b_output', biases['b_output'])


# In[12]:

# ========================================= 前向传播 =========================================
padding_type = 'SAME'
stride = [1, 1, 1, 1]
# keep = lambda _re, x: x if not _re else _re

def conv_basic(_input, _Weight, _bias, _reservation=None):
    # if not _reservation:
    #     _reservation = 0.7
    with tf.name_scope('Conv_1'):
        # reshape成思维的[batchSize , h, w , channel]
        __input = tf.reshape(_input, shape=[-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
        print('_input :::',_input.shape)
        # ___input = tf.nn.dropout(__input, keep(_reservation, 0.95), name='conv_1D')
        # stride = 2, 每移动2步
        conv2d_1 = tf.nn.conv2d(__input, _Weight['wc_1'], strides=stride, padding=padding_type)
        conv_1 = tf.nn.relu(tf.nn.bias_add(conv2d_1, _bias['bc_1']), name='conv_1')
        # pooling=2, pooling_strides=2
        pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding_type, name='pool_1')
        conv_1D = tf.nn.dropout(pool_1, keep(_reservation, 0.9), name='conv_1D')
        log_p('conv_1D', conv_1D.shape)
        
    with tf.name_scope('Conv_2'):
        conv2d_2 = tf.nn.conv2d(conv_1D, _Weight['wc_2'], strides=stride, padding=padding_type)
        conv_2 = tf.nn.relu(tf.nn.bias_add(conv2d_2, _bias['bc_2']), name='conv_2')
        pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding_type, name='pool_2')
        conv_2D = tf.nn.dropout(pool_2, keep(_reservation, 0.875), name='conv_2D')
        log_p('conv_2D', conv_2D.shape)

    with tf.name_scope('Conv_3'):
        conv2d_3 = tf.nn.conv2d(conv_2D, _Weight['wc_3'], strides=stride, padding=padding_type)
        conv_3 = tf.nn.relu(tf.nn.bias_add(conv2d_3, _bias['bc_3']), name='conv_3')
        pool_3 = tf.nn.max_pool(conv_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding_type, name='pool_3')
        conv_3D = tf.nn.dropout(pool_3, keep(_reservation, 0.85), name='conv_3D')
        log_p('conv_3D', conv_3D.shape)

    # with tf.name_scope('Conv_4'):
    #     conv2d_4 = tf.nn.conv2d(conv_3D, _Weight['wc_4'], strides=stride, padding=padding_type)
    #     conv_4 = tf.nn.relu(tf.nn.bias_add(conv2d_4, _bias['bc_4']), name='conv_4')
    #     pool_4 = tf.nn.max_pool(conv_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding_type, name='pool_4')
    #     conv_4D = tf.nn.dropout(pool_4, keep(_reservation, 0.825), name='conv_4D')
    #     log_p('conv_4D', conv_4D.shape)
    #
    # with tf.name_scope('Conv_5'):
    #     conv2d_5 = tf.nn.conv2d(conv_4D, _Weight['wc_5'], strides=stride, padding=padding_type)
    #     conv_5 = tf.nn.relu(tf.nn.bias_add(conv2d_5, _bias['bc_5']), name='conv_5')
    #     pool_5 = tf.nn.max_pool(conv_5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding_type, name='pool_5')
    #     conv_5D = tf.nn.dropout(pool_5, keep(_reservation, 0.8), name='conv_5D')
    #     log_p('conv_5D', conv_5D.shape)


    with tf.name_scope('Full_Connect_1'):
        # 需要知道当前 卷积后是多少的可以打开
        print('conv_5D ::::',conv_3D.shape)
        print('fc :::: ',_Weight['w_fc_1'].get_shape().as_list()[0])
        _fc_input = tf.reshape(conv_3D, [-1, _Weight['w_fc_1'].get_shape().as_list()[0]])
        full_connect_1 = tf.nn.relu(tf.add(tf.matmul(_fc_input, _Weight['w_fc_1']), _bias['b_fc_1']), name='full_connect_1')
        full_connect_1D = tf.nn.dropout(full_connect_1, keep(_reservation, 0.5), name='full_connect_1D')
    
    with tf.name_scope('Full_Connect_2'):
        full_connect_2 = tf.nn.relu(tf.add(tf.matmul(full_connect_1D, _Weight['w_fc_2']), _bias['b_fc_2']), name='full_connect_2')
        full_connect_2D = tf.nn.dropout(full_connect_2, keep(_reservation, 0.8), name='full_connect_2D')

    # with tf.name_scope('full_connect_3'):
    #     full_connect_3 = tf.nn.relu(tf.add(tf.matmul(full_connect_2D, _Weight['w_fc_3']), _bias['b_fc_3']), name='full_connect_3')
    #     full_connect_3D = tf.nn.dropout(full_connect_3, keep(_reservation, 1.0), name='full_connect_3D')

    with tf.name_scope('output'):
        # 最后这层是output 完成前向传播, 唔需RELU
        # output = tf.nn.sigmoid(tf.add(tf.matmul(full_connect_2D, _Weight['w_output']), _bias['b_output'], name='output'))
        output = tf.add(tf.matmul(full_connect_2D, _Weight['w_output']), _bias['b_output'], name='output')
        # output = tf.nn.softmax(tf.add(tf.matmul(full_connect_2D, _Weight['w_output']), _bias['b_output'], name='output'))
        log_p('output', output)
    print('------>forward output. -- DONE')
    # actv =
    return output



# In[13]:

# ========================================= 反向传播 =========================================
# __batch_size = 10
x = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], name='input_x')
y = tf.placeholder(tf.int64,  name='input_y')
# 将[batch_size,], 转换为[batch_size, n_output]
_y = tf.one_hot(indices=y, depth=n_output, off_value=0.0, on_value=1.0)
# 如果y的值 = 0，则log(y)则会出错，解决方法是，将上式修改为：
# print(_y)

reservation = tf.placeholder(tf.float32, name='reservation')

pred = conv_basic(x, Weights, biases, reservation)
# _pred = tf.clip_by_value(pred,1e-10,1.0)
print(pred)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=_y), name='loss')
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer').minimize(loss, name='minimize_loss')
with tf.name_scope('across'):
    across = tf.equal(tf.argmax(pred, 1), tf.argmax(_y, 1))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(across, tf.float32))
print('------>backward calculator. -- DONE')

# In[ ]:

# ========================================= 训练启动 =========================================
# step4 --------------------trainRUN---------------
training_epochs = 50 # 200
display_step = 10 # (4172/20) ~= 200   (1190/20) ~= 59    200/59 ~= 3 但还是改为5次吧
batch_size = 30 # 20
capacity = 1000 + 3 * batch_size
# batch_size_test = 20
r_rate = 1.0




saver = tf.train.Saver()

with tf.Session() as sess:
    try:
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('Logs/plane', sess.graph)
        image_tr, label_tr = inputs(RECORD_TRAIN, batch_size=batch_size)
        image_ts, label_ts = inputs(RECORD_TEST, batch_size=batch_size)

        # with tf.Graph().as_default():
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess)
        counter = 0

        for epoch in range(training_epochs):
            avg_loss = 0
            counter += 1
            batch_xs_train, batch_ys_train = sess.run([image_tr, label_tr])

            batch_xs_test, batch_ys_test = sess.run([image_ts, label_ts])
            # 可能数据本身存在Nan的情况
            # assert np.any(np.isnan(batch_xs_train))
            # assert np.any(np.isnan(batch_ys_train))
            # assert np.any(np.isnan(batch_xs_test))
            # assert np.any(np.isnan(batch_ys_test))
            # batch_ys__train = tran_two_d(np.array(batch_ys_train).reshape(batch_size, 1), dont=False)
            # batch_ys__test = tran_two_d(np.array(batch_ys_test).reshape(batch_size, 1), dont=False)
            batch_ys__train = batch_ys_train
            batch_ys__test = batch_ys_test

            for i in range(batch_size):
                sess.run(optimizer, feed_dict={x: batch_xs_train, y: batch_ys__train, reservation:0.75})
                avg_loss += sess.run(loss,feed_dict={x: batch_xs_train, y: batch_ys__train, reservation: r_rate}) / batch_size

            if epoch % display_step == 0:
                log_p('batch_ys__train',batch_ys__train)
                log_p('batch_ys__test', batch_ys__test)
                print("Epoch:%03d/%03d -- avg_loss: %.9f" % (epoch, training_epochs, avg_loss))

                train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs_train, y: batch_ys__train, reservation: r_rate})
                test_accuracy = sess.run(accuracy, feed_dict={x: batch_xs_test, y: batch_ys__test, reservation: r_rate})
                print("Train accuracy: %.3f And Test accuracy: %.3f " % (train_accuracy, test_accuracy))

                loss_acc = sess.run(loss, feed_dict={x: batch_xs_train, y: batch_ys__train, reservation: r_rate})
                pred_acc = sess.run(pred, feed_dict={x: batch_xs_train, y: batch_ys__train, reservation: r_rate})
                across_acc = sess.run(across, feed_dict={x: batch_xs_train, y: batch_ys__train, reservation: r_rate})
                print("pred_acc:{a}, across_acc:{b} , loss_acc:{c}".format(a=pred_acc, b=across_acc,c=loss_acc))

            summary = sess.run(merged, feed_dict={x: batch_xs_train, y: batch_ys__train, reservation: 0.7})
            writer.add_summary(summary, epoch)

            # 每50步保存
            if counter % 50 == 0:
                saver.save(sess, MODEL_SAVEPATH)

        coord.request_stop()
        coord.join(threads,stop_grace_period_secs=30, ignore_live_threads=True) # 放弃那些自动
    except tf.errors.OutOfRangeError:
        print("OutOfRangeError end!")

    # except Exception as e:
    #     # 可能还存在啥子错误
    #     print('Exception:',e)
    #     coord.stop_on_exception()
    finally:

        saver.save(sess, MODEL_SAVEPATH_LAST)
        print("RUN DONE")

