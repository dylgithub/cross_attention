#encoding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import GRUCell
'''
做语义相似度时获得句子编码，使用交叉Attention，同时补充一些特征
具体可参考：https://blog.csdn.net/weixin_38526306/article/details/88045134
'''
def Bi_GRU(inputs):
    cell_fw = GRUCell(200)
    cell_bw = GRUCell(200)
    (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,inputs,dtype=tf.float32)
    return tf.concat([output_fw_seq, output_bw_seq], axis=-1)
def cross_attention(input1,input2):
    '''
    :param input1: [b,s1,h1]
    :param input2: [b,s2,h2]
    :return:
    '''
    input1=tf.convert_to_tensor(input1,dtype=tf.float32)
    input2=tf.convert_to_tensor(input2,dtype=tf.float32)
    #这里的维度为[b,s1,s2]
    #每一行的值是句子s1中每个字对句子s2中每个字的关注度
    #每一列的值是句子s2中每个字对句子s1中每个字的关注度
    attention=tf.matmul(input1,tf.transpose(input2,[0,2,1]))
    #分别对s1中字对s2中各个字的关注进行归一化
    w_att_1=tf.nn.softmax(attention,axis=-1)
    # 分别对s2中字对s1中各个字的关注进行归一化
    w_att_2=tf.nn.softmax(tf.transpose(attention,[0,2,1]),axis=-1)
    # 这里的维度为[b,s1,h2]
    # 得到的是s1对s2的关注度
    align1=tf.matmul(w_att_1,input2)
    # 这里的维度为[b,s2,h1]
    # 得到的是s2对s1的关注度
    align2=tf.matmul(w_att_2,input1)
    #随后对多种特征进行合并
    # 拼接align1和align2，若相互的关注度点相同则更相似
    # input1-align1是计算本身和关注另一个句子的差异性，更利于相似度的计算
    concat1=tf.concat([input1,align1,input1-align1,input1*align1],axis=-1)
    concat2=tf.concat([input2,align2,input2-align2,input2*align2],axis=-1)
    # concat1 = tf.concat([align1, input1 - align1], axis=-1)
    # concat2 = tf.concat([align2, input2 - align2], axis=-1)
    print(concat1)
    with tf.variable_scope('gru1'):
        concat1=Bi_GRU(concat1)
    with tf.variable_scope('gru2'):
        concat2=Bi_GRU(concat2)
    #获得全局和局部特征
    max_pool1=tf.reduce_max(concat1,axis=1)
    avg_pool1=tf.reduce_mean(concat1,axis=1)

    max_pool2 = tf.reduce_max(concat2, axis=1)
    avg_pool2 = tf.reduce_mean(concat2, axis=1)

    final1=tf.concat([max_pool1,avg_pool1],axis=-1)
    final2=tf.concat([max_pool2,avg_pool2],axis=-1)
    print(final1)
    print(final2)
if __name__ == '__main__':
    input1=np.random.standard_normal([64,200,768])
    input2=np.random.standard_normal([64,200,768])
    cross_attention(input1,input2)