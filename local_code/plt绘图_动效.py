# -*- coding:utf-8 -*-
__author = ''
# -*- coding:utf-8 -*-
# __classOf 封装神经层功能 x_data^2 - 0.5 + noise

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
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 根据in_size,out_size, 随机生成变量,Weights可为矩阵
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # 建议不要为0
    # out_size , 决定了这个是几列的变量, biases最多为列
    Wx_plus_b = tf.matmul(input, Weights) + biases
    # W*x + n
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

x_data_feed = tf.placeholder(tf.float32, [None, 1])
y_data_feed = tf.placeholder(tf.float32, [None, 1])
# None是接收任意数量

# 模型建立为 inL - hidL - outL   1-10-1
layer_1 = add_layer(x_data_feed, 1, 10, activation_func=tf.nn.relu)
# 这一层是hiddenLayer, 上层是1,假设本层为10个神经元,激励函数为RULE
prediction = add_layer(layer_1, 10, 1)
# 这一层是outputLayer,接口数据来自layer_1, 上层是10,本层为1,无激励函数 输出y^

pre_loss_sum = tf.reduce_sum(tf.square(y_data_feed - prediction), reduction_indices=[1])
# reduction_indices 关键参数, 似乎是相对上层的np.newaxis #y_data - y 之后的平方, 总数
loss = tf.reduce_mean(pre_loss_sum)
# 求均值,为loss

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
# 优化器以0.1的效率, 对误差Loss进行提取修正 学习率与学习次数, 必须是高反比. 学习率越低, 学习次数必须越大!

# mark ------------------------------------------ dataCore ----------------------- end

# mark ------------------------------------------ 数据可视化 -----------------------  start

fig = plt.figure()
# 初始化画板
ax = fig.add_subplot(1, 1, 1)  # 固定?
ax.scatter(x_data, y_data)
# 展示x,y
# plt.ion()  #不允许阻塞 本地代码有效
# plt.show()   #开始绘制 本地代码的时候打开,会动图


# mark ------------------------------------------ 数据可视化 -----------------------  end



init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)  # 激活数据
    for step in xrange(30000):
        # 0.0028274
        session.run(train_step, feed_dict={x_data_feed: x_data, y_data_feed: y_data})
        if step % 50 == 0:
            #             print session.run(loss,feed_dict={x_data_feed:x_data,y_data_feed:y_data})
            try:
                ax.lines.remove(lines[0])
                # 移除前一张图片,再绘制一张新的
            except:
                pass
            prediction_value = session.run(prediction, feed_dict={x_data_feed: x_data})
            # 因为上方, prediction需要x_data, 所以给予x_data
            print  prediction_value
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            # 红色, 线宽5
            plt.pause(0.05)
            # 每次暂停0.1s

# 如果是notebook, 不会启动plt,直接展示
plt.show()  # 开始绘制
 