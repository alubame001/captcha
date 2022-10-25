# /usr/bin/env python
# coding=utf-8
from gen_captcha import gen_captcha_text_and_image, MAX_CAPTCHA
from constants import number, alphabet, ALPHABET, MATH_STRINGS, CHAR_2_POS_DICT, POS_2_CHAR
from gen_captcha import modelNo
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160

char_set=number
CHAR_SET_LEN = len(char_set)

def convert2gray(imag_):
    if len(imag_.shape) > 2:  
        gray = np.mean(imag_, -1)  #按照最后一个维度求均值
        return gray  
    else:  
        return imag_
    
#将文本转化为向量
def text2vec(text): 
    text_len = len(text)  
    if text_len > MAX_CAPTCHA:  
        raise ValueError   #('验证码最长4个字符')  
    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)  
    for key,val in enumerate(text):    #enumerate(text)同时返回列表的索引和值
        idx = key * 10 + int(val)  
        vector[idx] = 1  
    return vector  
  

#将向量转化为文本
def vev2text(vec):
    text=[]
    char_pos = vec.nonzero()[0]    #np.nonzero返回非零值的索引
    for i in enumerate(char_pos):  
        number = i % 10             #通过取余数的办法获得数字值
        text.append(str(number))          
    return "".join(text)  
    
#生成训练集
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH]) 
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])
    
    #有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        ''' 获取一张图，判断其是否符合（60，160，3）的规格'''
        while True:  
            text, image = gen_captcha_text_and_image()  
            if image.shape == (60, 160, 3):  
                return text, image  
   
    for i in range(batch_size):  
        text, image = wrap_gen_captcha_text_and_image()  
        image = convert2gray(image)  
        #将图片数组一维化，即扁平化，再将像素点压缩在0~1之间
        batch_x[i,:] = image.flatten() / 255 # (image.flatten()-128)/128  mean为0  
        #同时将文本也对应同时将文本也对应在两个二维组的同一行
        batch_y[i,:] = text2vec(text)
        
    return batch_x, batch_y  
    
#占位 
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32) # dropout

#正向传播：
#参数的初始化过程中需要加入一小部分的噪声以破坏参数整体的对称性，同时避免梯度为0，即w_alpha=0.01, b_alpha=0.1
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):  
    #对输入的图像进行预处理：转换为tensorflow支持的格式：
    #[batch, in_height, in_width, in_channels] ，-1表示自动计算
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])  
    
    #strides为步长，格式为：[filter_height, filter_width, in_channels, out_channels]
    w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))  
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))  
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))  
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    conv1 = tf.nn.dropout(conv1, keep_prob)  
   
    w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))  
    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))  
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))  
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    conv2 = tf.nn.dropout(conv2, keep_prob)  
   
    w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))  
    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))  
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))  
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    conv3 = tf.nn.dropout(conv3, keep_prob)  
   
    # Fully connected layer  
    w_d = tf.Variable(w_alpha*tf.random_normal([8*20*64, 1024]))  
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]))  
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])  
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))  
    dense = tf.nn.dropout(dense, keep_prob)  
   
    w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))  
    b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))  
    out = tf.add(tf.matmul(dense, w_out), b_out)   
    return out  
# 反向传播 
def train_crack_captcha_cnn(): 
    start_time = time.time()
    output = crack_captcha_cnn()
    #损失函数
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output,labels=Y))  
    #优化器：
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss) 
    #将最终的输出值转化为三维数组，高为4，宽为10，第三个维度（batch）自动计算，
    #每个二维数组的每一行就代表验证码每一位数字10各类别的概率值
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])  
    #按照第三个维度索取每一个二维数组取得最大值的索引
    max_idx_p = tf.argmax(predict, 2) 
    #同时对真实值Y也索取其最大值的索引
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)  
    #计算精确度
    correct_pred = tf.equal(max_idx_p, max_idx_l)  
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) 
    
    #存储训练的数据    
    saver = tf.train.Saver(max_to_keep=1) 
    
    #创建计算视图
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  
        #max_acc = 0
        step = 0  
        #用while循环迭代，直到精确度大于某一阈值，用for循环可以找到规定循环次数内的最高精度
        while True: 
            #每次训练选取128个样本
            batch_x, batch_y = get_next_batch(128)  
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})  
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), step, loss_)
              
            # 每10 step计算一次准确率  
            if step % 100 == 0: 
                #每次选取128个样本用来验证
                batch_x_test, batch_y_test = get_next_batch(128)  
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})  
                print(u'***************************************************************第%s次的准确率为%s' % (step, acc))  
                #当精确度高于55%时就保存模型
                if acc > 0.98:  
                    saver.save(sess, "models/" + str(modelNo) + "/" + str(modelNo) + ".model", global_step=step)
                    print(time.time() - start_time)
                    break
   
            step += 1
            
            
train_crack_captcha_cnn()