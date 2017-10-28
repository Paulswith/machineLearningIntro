# -*- coding:utf-8 -*-
__author = ''
# -*- coding:utf-8 -*-
# __classOf tensorboard画图 x_data^2 - 0.5 + noise

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt  # py可视化


def add_layer(input, in_size, out_size, activation_func=None):
    """
    input -> X
    in_size -> 上层的神经元数量
    out_size -> 本层的神经元数量
    activation_func -> 激励函数
    """
    with tf.name_scope('layer'):
        # tfb-> 集合到layer
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='Weight')
            # 根据in_size,out_size, 随机生成变量,Weights可为矩阵
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='bias')  # 建议不要为0
            # out_size , 决定了这个是几列的变量, biases最多为列
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(input, Weights) + biases
            # W*x + n
        ## activation_func 传递进来的时候默认就有tfb
        if activation_func:
            outputs = activation_func(Wx_plus_b)
        else:
            outputs = Wx_plus_b
        return outputs


# mark ------------------random_get_data ----------------
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# print  x_data #生成大小为300的数组, 值取(-1,1). 行不变, 更改列的维度,newaxist用法
noise = np.random.normal(0, 0.05, x_data.shape)
# print noise #初始值为0, 方差为0.05, 维度同步x_data 方差-> 至少差距在nn
y_data = np.square(x_data) - 0.5 + noise
#                          x_data^2 - 0.5 + noise



# mark ------------------------------------------ dataCore ----------------------- start
with tf.name_scope('inputs'):
    # tfb-> 集合子item, 并起名字
    x_data_feed = tf.placeholder(tf.float32, [None, 1], name='x_input')
    y_data_feed = tf.placeholder(tf.float32, [None, 1], name='y_input')
    # None是接收任意数量

# 模型建立为 inL - hidL - outL   1-10-1
layer_1 = add_layer(x_data_feed, 1, 10, activation_func=tf.nn.relu)
# 这一层是hiddenLayer, 上层是1,假设本层为10个神经元,激励函数为RULE
prediction = add_layer(layer_1, 10, 1)
# 这一层是outputLayer,接口数据来自layer_1, 上层是10,本层为1,无激励函数

with tf.name_scope('loss'):
    pre_loss_sum = tf.reduce_sum(tf.square(y_data_feed - prediction), reduction_indices=[1])
    # reduction_indices 关键参数, 似乎是相对上层的np.newaxis #y_data - y 之后的平方, 总数
    loss = tf.reduce_mean(pre_loss_sum)
    # 求均值,为loss
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
# 优化器以0.1的效率, 对误差Loss进行提取修正 学习率与学习次数, 必须是高反比. 学习率越低, 学习次数必须越大!

# mark ------------------------------------------ dataCore ----------------------- end


init = tf.global_variables_initializer()
with tf.Session() as session:
    #     writer = tf.train.FileWriter('logs/',session.graph)
    writer = tf.summary.FileWriter('logs/', session.graph)
    session.run(init)  # 激活数据
    # 执行结束后,终端查看, tensorboard --logdir='logs/'  映射到本地端口查看
 