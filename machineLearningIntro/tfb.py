# -*- coding:utf-8 -*-
__author = ''
# __classOf tensorboard画图 x_data^2 - 0.5 + noise

import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt  # py可视化


def add_layer(input, layer_name, in_size, out_size, activation_func=None):
    """
    input -> X
    in_size -> 上层的神经元数量
    out_size -> 本层的神经元数量
    [insize , outsize] // 行列
    activation_func -> 激励函数
    """
    with tf.name_scope('layer'):
        # tfb-> 集合到layer
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='Weight')
            # 根据传递进来的,[insize , outsize]  生成随机数
            tf.summary.histogram(layer_name + '/Weights', Weights)
            # 想要收集的变量
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='bias')  # 建议不要为0
            #biases 只为行向量
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(input, Weights) + biases
            #matmul 乘法
            tf.summary.histogram(layer_name + '/Wx_plus_b', Wx_plus_b)
        if activation_func:
            outputs = activation_func(Wx_plus_b)
        else:
            outputs = Wx_plus_b
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs


# mark ------------------random_get_data ----------------
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# mark ------------------------------------------ dataCore ----------------------- start
with tf.name_scope('inputs'):
    x_data_feed = tf.placeholder(tf.float32, [None, 1], name='x_input')
    y_data_feed = tf.placeholder(tf.float32, [None, 1], name='y_input')

layer_1 = add_layer(x_data_feed, 'hidLayer', 1, 10, activation_func=tf.nn.relu)
prediction = add_layer(layer_1, 'outLayer', 10, 1)

with tf.name_scope('loss'):
    # loss
    pre_loss_sum = tf.reduce_sum(tf.square(y_data_feed - prediction), reduction_indices=[1])
    loss = tf.reduce_mean(pre_loss_sum)
    tf.summary.scalar('loss', loss)  # 特殊收集方式

with tf.name_scope('train_step'):
    #激励函数 学习率
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# mark ------------------------------------------ dataCore ----------------------- end




with tf.Session() as session:

    init = tf.global_variables_initializer()
    merged = tf.summary.merge_all()
    # 集合上方的数据
    writer = tf.summary.FileWriter('logs/', session.graph)
    session.run(init)  # 激活数据

    for step in xrange(1000):
        session.run(train_step, feed_dict={x_data_feed: x_data, y_data_feed: y_data})
        if step % 50 == 0:
            result = session.run(merged, feed_dict={x_data_feed: x_data, y_data_feed: y_data})
            # 执行收集
            writer.add_summary(result, step)
            # 添加到writer里面,必须跟随step, 相当于x轴


 