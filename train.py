import os, sys, time, random
import pandas as pd
import numpy as np
import tensorflow as tf
from model import Model

# from tensorflow import ConfigProto
# from tensorflow import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

#Note: this code must be run using tensorflow 1.4.0

class DataInput:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration
        ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
        self.i += 1
        u, hist, i, y ,g,a,n= [], [], [], [],[],[],[]
        for t in ts:
            u.append(t[0])
            hist.append(t[1])
            i.append(t[2])
            y.append(t[3])
            g.append(t[4])
            a.append(t[5])
            n.append(t[6])
            #genres.append(t[6])
        return self.i, (u, hist, i, y,g,a,n) #迭代次数+四元组

def test(sess, model, test_set):
    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    arr = []
    for _, uij in DataInput(test_set, batch_size):
        score, label, user, _, _, _ , _ = model.test(sess, uij) #调用model里的test
        for index in range(len(score)):
            if label[index] > 0:
                arr.append([0, 1, score[index]])
            elif label[index] == 0:
                arr.append([1, 0, score[index]])
    arr = sorted(arr, key=lambda d:d[2])
    for record in arr:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2
    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5
    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None

def hit_rate(sess, model, test_set):
    hit, arr = [], []
    userid = list(set([x[0] for x in test_set]))
    for _, uij in DataInput(test_set, batch_size):
        score, label, user, _, _,_,_ = model.test(sess, uij)
        for index in range(len(score)):
            if score[index] > 0.5:
                arr.append([label[index], 1, user[index]])
            else:
                arr.append([label[index], 0, user[index]])
    for user in userid:
        arr_user = [x for x in arr if x[2]==user and x[1]==1]
        if len(arr_user)==0:
            continue
        else:
            hit.append(sum([x[0] for x in arr_user])/len(arr_user))
    return np.mean(hit)

def coverage(sess, model, test_set):
    rec_item = []
    for _, uij in DataInput(test_set, batch_size):
        score, label, user, item, _,_,_ = model.test(sess, uij)
        for index in range(len(score)):
            if score[index] > 0.5:
                rec_item.append(item[index])
    return len(set(rec_item)) / len(itemid)

def unexpectedness(sess, model, test_set):
    unexp_list = []
    for _, uij in DataInput(test_set, batch_size):
        score, label, user, item, unexp ,_,_= model.test(sess, uij)
        for index in range(len(score)):
            unexp_list.append(unexp[index])
    return np.mean(unexp_list)

random.seed(625)
np.random.seed(625)
tf.set_random_seed(625)
batch_size = 512
#数据预处理
# data = pd.read_csv('test.txt', names=['utdid','vdo_id','click','hour'])
data = pd.read_csv('m1.csv', names=['utdid','vdo_id','click','hour','gender','age','movie_name'],usecols=[0,1,2,3,4,5,7],header=1)

#data = pd.read_csv('train_data_r01.csv', names=['click','utdid','vdo_id','item_name','genre','tags','actor_display','writer_display','region','hour'])
user_id = data[['utdid']].drop_duplicates().reindex()
user_id['user_id'] = np.arange(len(user_id))
data = pd.merge(data, user_id, on=['utdid'], how='left')
item_id = data[['vdo_id']].drop_duplicates().reindex()
item_id['video_id'] = np.arange(len(item_id))
data = pd.merge(data, item_id, on=['vdo_id'], how='left')
data = data[['user_id','video_id','click','hour','gender','age','movie_name']]
print(data)
userid = list(set(data['user_id']))
ageid=list(set(data['age']))
#genres_lens=len(np.max(data["genres"]))
#print(genres_lens)
age_count=len(ageid)+1
itemid = list(set(data['video_id']))
user_count = len(userid)
print(user_count)
item_count = len(itemid)
print(item_count)
#划分数据集

validate = 4 * len(data) // 5
train_data = data.loc[:validate,]
#print(train_data)
test_data = data.loc[validate:]
#other_data =data.loc[2*validate:]
train_set, test_set = [], []

for user in userid:
    train_user = train_data.loc[train_data['user_id']==user]
    train_user = train_user.sort_values(['hour'])
    length = len(train_user)
    train_user.index = range(length)
    if length > 10:
        for i in range(length-10):
            train_set.append((train_user.loc[i+9,'user_id'],list(train_user.loc[i:i+9,'video_id']), train_user.loc[i+9,'video_id'], float(train_user.loc[i+9,'click']), train_user.loc[i+9,'gender'],train_user.loc[i+9,'age'],eval(train_user.loc[i+9,'movie_name']),))
    test_user = test_data.loc[test_data['user_id']==user]
    test_user = test_user.sort_values(['hour'])
    length = len(test_user)
    test_user.index = range(length)
    if length > 10:
        for i in range(length-10):
            test_set.append((test_user.loc[i+9,'user_id'], list(test_user.loc[i:i+9,'video_id']),test_user.loc[i+9,'video_id'], float(test_user.loc[i+9,'click']),test_user.loc[i+9,'gender'],test_user.loc[i+9,'age'],eval(test_user.loc[i+9,'movie_name']),))
random.shuffle(train_set)
print("21324")
random.shuffle(test_set)
train_set = train_set[:len(train_set)//batch_size*batch_size]

#train_set.to_csv("read1",index=False)
test_set = test_set[:len(test_set)//batch_size*batch_size]
print(test_set)

#调用session执行
gpu_options = tf.GPUOptions(allow_growth=True)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) as sess:
    model = Model(user_count, item_count,age_count ,batch_size)
    sess.run(tf.global_variables_initializer())#初始化参数的过程（赋值），保存于saver中
    sess.run(tf.local_variables_initializer()) #不在saver中保存的

    print('test_auc: %.4f' % test(sess, model, test_set))
    sys.stdout.flush() #间隔输出
    lr = 1
    start_time = time.time()
    last_auc = 0.0

    for _ in range(40):  #epoch=1000？？√
        random.shuffle(train_set) #打乱顺序
        epoch_size = round(len(train_set) / batch_size)
        loss_sum = 0.0
        for _, uij in DataInput(train_set, batch_size):
            loss = model.train(sess, uij, lr)
            loss_sum += loss
            if model.global_step.eval() % 100 == 0:
                auc = test(sess, model, test_set)
                train_auc = test(sess, model, train_set)
                print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_AUC: %.4f\tTrain_AUC: %.4F' %
                      (model.global_epoch_step.eval(), model.global_step.eval(),loss_sum / 1000, auc, train_auc))
                sys.stdout.flush()
                loss_sum = 0.0
        print('Epoch %d DONE\tCost time: %.2f' %
              (model.global_epoch_step.eval(), time.time()-start_time))
        if abs(train_auc - last_auc) < 0.001:
            lr = lr / 2
        last_auc = train_auc
        sys.stdout.flush()
        model.global_epoch_step_op.eval()
        hit = hit_rate(sess, model, test_set)
        cov = coverage(sess, model, test_set)
        unexp = unexpectedness(sess, model, test_set)
        print('Epoch %d Eval_Hit_Rate: %.4f' % (model.global_epoch_step.eval(), hit))
        print('Epoch %d Eval_Coverage: %.4f' % (model.global_epoch_step.eval(), cov))
        print('Epoch %d Eval_Unexpectedness: %.4f' % (model.global_epoch_step.eval(), unexp))