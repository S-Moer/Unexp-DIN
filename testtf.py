import tensorflow as tf;
import numpy as np;
import os, sys, time, random
import pandas as pd
import numpy as np
import tensorflow as tf
from model import Model


# a1 = tf.global_variables_initializer(name='a1', shape=[2, 3], initializer=tf.random_normal_initializer(mean=0, stddev=1))
# a2 = tf.global_variables_initializer(name='a2', shape=[50], initializer=tf.constant_initializer(0.0))
# a3 = tf.global_variables_initializer(name='a3', shape=[2, 3], initializer=tf.ones_initializer())
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print
#     sess.run(a1)
#     print
#     sess.run(a2)
#     print
#     sess.run(a3)


# current=np.array([
#         [0,7,1,2,2],
#         [1,7,3,4,3],
#         [2,7,5,6,6],
#         [3,7,7,8,7],
#         [4,7,7,8,7],
#         [5,7,7,8,7]
# ])
#
# points_e = tf.expand_dims(current, axis=1)
# print(points_e.shape)
# with tf.Session() as sess:
#     print
#     sess.run(points_e)

data = pd.read_csv('test.txt', names=['utdid','vdo_id','click','hour'])
user_id = data[['utdid']].drop_duplicates().reindex()
user_id['user_id'] = np.arange(len(user_id))
data = pd.merge(data, user_id, on=['utdid'], how='left')
item_id = data[['vdo_id']].drop_duplicates().reindex()
item_id['video_id'] = np.arange(len(item_id))
data = pd.merge(data, item_id, on=['vdo_id'], how='left')
data = data[['user_id','video_id','click','hour']]
userid = list(set(data['user_id']))
itemid = list(set(data['video_id']))
user_count = len(userid)
item_count = len(itemid)
#划分数据集
validate = 4 * len(data) // 5
train_data = data.loc[:validate,]
test_data = data.loc[validate:,]
train_set, test_set = [], []

for user in userid:
    train_user = train_data.loc[train_data['user_id']==user]
    train_user = train_user.sort_values(['hour'])
    length = len(train_user)
    train_user.index = range(length)
    if length > 10:
        for i in range(length-10):
            train_set.append((train_user.loc[i+9,'user_id'], list(train_user.loc[i:i+9,'video_id']), train_user.loc[i+9,'video_id'], float(train_user.loc[i+9,'click'])))
    test_user = test_data.loc[test_data['user_id']==user]
    test_user = test_user.sort_values(['hour'])
    length = len(test_user)
    test_user.index = range(length)
    if length > 10:
        for i in range(length-10):
            test_set.append((test_user.loc[i+9,'user_id'], list(test_user.loc[i:i+9,'video_id']), test_user.loc[i+9,'video_id'], float(test_user.loc[i+9,'click'])))
# random.shuffle(train_set)
# random.shuffle(test_set)
train_set = train_set[:len(train_set)//32*32]
test_set = test_set[:len(test_set)//32*32]
# s = '1+2+3*5-2'
# print(s.eval())
print(train_set)
print(test_set)
