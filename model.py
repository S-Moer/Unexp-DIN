import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

class Model(object):
    def __init__(self, user_count, item_count, age_count,batch_size):
        hidden_size = 256 #向量维度
        hidden_size2=64
        long_memory_window = 10
        short_memory_window = 3
        window_sizes = {2, 3, 4, 5}

        #设置形参占位符，执行的时候再赋值，（dtype，shape，name）
        self.u = tf.placeholder(tf.int32, [batch_size,]) # [B]
        self.i = tf.placeholder(tf.int32, [batch_size,]) # [B]

        self.movie_titles = tf.placeholder(tf.int32, [batch_size, 15], name="movie_titles")

        self.y = tf.placeholder(tf.float32, [batch_size,]) # [B]
        self.hist = tf.placeholder(tf.int32, [batch_size, long_memory_window]) # [B, T]
        #self.genres=tf.placeholder(tf.int32, [batch_size, 18])
        #movie_categories = tf.placeholder(tf.int32, [None, 18], name="movie_categories")
        self.g = tf.placeholder(tf.int32, [batch_size, ])  # [B]
        self.a = tf.placeholder(tf.int32, [batch_size, ])
        print(self.hist)
        self.lr = tf.placeholder(tf.float64, [])


        #
        user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_size // 4])
        print(user_emb_w)#6040,64
        #genres_emb_w=tf.get_variable("genres_emb_w",[18,hidden_size//4])
        gender_emb_w = tf.get_variable("gender_emb_w", [2, hidden_size // 8])
        age_emb_w=tf.get_variable("age_emb_w",[age_count,hidden_size//4])
        item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_size // 4])
        print(item_emb_w)#3706,64
        #初始化，值全部是常量0.0
        user_b = tf.get_variable("user_b", [user_count], initializer=tf.constant_initializer(0.0),)
        print(user_b)#6040
        #genres_b = tf.get_variable("genres_b", [18], initializer=tf.constant_initializer(0.0), )
        gender_b = tf.get_variable("gender_b", [2], initializer=tf.constant_initializer(0.0), )
        age_b = tf.get_variable("age_b", [age_count], initializer=tf.constant_initializer(0.0), )
        item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0))
        print(item_b)#3706
        #两个拼接g
        #genres_emb=tf.nn.embedding_lookup(genres_emb_w,self.genres)
        #print(genres_emb)#32,18,32
        #genres_emb=tf.reduce_sum(genres_emb,axis=1,)

        pool_layer_flat, dropout_layer = self.get_movie_cnn_layer(self.movie_titles)
        print(pool_layer_flat,dropout_layer)

        dropout_layer=tf.reduce_sum(dropout_layer,axis=1)
        print(pool_layer_flat, dropout_layer)
        #print(dropout_layer)
        print("729889`638746")
        #print(genres_emb)#32,32
        item_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.i),
            tf.nn.embedding_lookup(user_emb_w, self.u),
            tf.nn.embedding_lookup(gender_emb_w, self.g),
            tf.nn.embedding_lookup(age_emb_w, self.a),
            dropout_layer
            ], axis=1)#item+user
        print(item_emb)#32,128
        #获取一个i/u的切片,按维度取
        item_b = tf.gather(item_b, self.i)
        print(item_b)#32
        gender_b=tf.gather(gender_b,self.g)
        age_b = tf.gather(age_b, self.a)
        user_b = tf.gather(user_b, self.u)
        #tf.nn.embedding_lookup(item_emb_w, tf.slice(self.movie_titles, [0, 0], [batch_size, 15])),
        #genres_b = tf.gather(genres_b,self.genres)
        print(user_b)#32
        #切片取出一部分？？
        h_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, tf.slice(self.hist, [0,0], [batch_size, long_memory_window])),
            tf.tile(tf.expand_dims(tf.nn.embedding_lookup(user_emb_w, self.u), 1), [1, long_memory_window, 1]),
            tf.tile(tf.expand_dims(tf.nn.embedding_lookup(gender_emb_w, self.g), 1), [1, long_memory_window, 1]),
            tf.tile(tf.expand_dims(tf.nn.embedding_lookup(age_emb_w, self.a), 1), [1, long_memory_window, 1]),#张量扩展
            tf.tile(tf.expand_dims(dropout_layer, 1), [1, long_memory_window, 1]),
            ], axis=2)#32,10,128
        print(h_emb)
        unexp_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, tf.slice(self.hist, [0,long_memory_window-short_memory_window], [batch_size, short_memory_window])),
            tf.tile(tf.expand_dims(tf.nn.embedding_lookup(user_emb_w, self.u), 1), [1, short_memory_window, 1]),
            tf.tile(tf.expand_dims(tf.nn.embedding_lookup(gender_emb_w, self.g), 1), [1, short_memory_window, 1]),
            tf.tile(tf.expand_dims(tf.nn.embedding_lookup(age_emb_w, self.a), 1), [1, short_memory_window, 1]),
            tf.tile(tf.expand_dims(dropout_layer, 1), [1, short_memory_window, 1]),
            ], axis=2) #32,3,128
        print(unexp_emb)
        h_long_emb = tf.nn.embedding_lookup(item_emb_w, tf.slice(self.hist, [0,0], [batch_size, long_memory_window]))  #给平均漂移做输入
        print(h_long_emb)#32,10,64
        h_short_emb = tf.nn.embedding_lookup(item_emb_w, tf.slice(self.hist, [0,long_memory_window-short_memory_window], [batch_size, short_memory_window]))
        print(h_short_emb)#32,3,64
        # Long-Short-Term User Preference
        #with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
        long_output, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=h_emb, dtype=tf.float32) #设置GRU的层数以及输入
        print(long_output)#32,10,128
        print(_)#32,128其中_就是权重的矩阵
        long_preference, _ = self.seq_attention(long_output, hidden_size, long_memory_window)  #放入attention net中
        print(long_preference)#32,128
        print(_)#32,10
        long_preference = tf.nn.dropout(long_preference, 0.1) #以0.1的概率抑制神经元32,128
        print("ss")
        print(long_preference)#32,128
        #short_output, _ = tf.nn.dynamic_rnn(GRUCell(hidden_size), inputs=unexp_emb, dtype=tf.float32)
        #short_preference, _ = self.seq_attention(short_output, hidden_size, long_memory_window)
        #short_preference = tf.nn.dropout(short_preference, 0.1)

        #点击率
        #Combine Long-Short-Term-User-Preferences
        concat = tf.concat([long_preference, item_emb], axis=1)
        print(concat)#32,256
        concat = tf.layers.batch_normalization(inputs=concat) #向量的处理和缩放
        print(concat)#32，256
        concat = tf.layers.dense(concat, 80, activation=tf.nn.sigmoid, name='f1') #定义三层的全连接
        print(concat)#32,80
        concat = tf.layers.dense(concat, 40, activation=tf.nn.sigmoid, name='f2')
        print(concat)#32,34
        concat = tf.layers.dense(concat, 1, activation=None, name='f3')
        print(concat)#32,1
        concat = tf.reshape(concat, [-1])
        print(concat)#32个数字就是点击率

        #Personalized & Contextualized Unexpected Factor
        unexp_factor = self.unexp_attention(item_emb, unexp_emb, [long_memory_window]*batch_size) #self-attention
        print(unexp_factor)#32,128
        unexp_factor = tf.layers.batch_normalization(inputs = unexp_factor)
        print(unexp_factor)#32,128
        unexp_factor = tf.reshape(unexp_factor, [-1, hidden_size])
        print(unexp_factor)#32,128
        unexp_factor = tf.layers.dense(unexp_factor, hidden_size) # MLP
        print(unexp_factor)#32,128
        unexp_factor = tf.layers.dense(unexp_factor, 1, activation=None)
        print(unexp_factor)#32,1
        #If we choose to use binary values
        #unexp_gate = tf.to_float(tf.reshape(unexp_gate, [-1]) > 0.5)
        unexp_factor = tf.reshape(unexp_factor, [-1]) #平铺
        print(unexp_factor)#32

        #Unexpectedness (with clustering of user interests)
        self.center = self.mean_shift(h_long_emb)
        print(self.center)#32,10,64
        unexp=self.center
        unexp, _ = self.cluster_attention(unexp, hidden_size2, long_memory_window)
        #unexp = tf.reduce_mean(self.center, axis=1) #计算平均值
        print(unexp)#32,64
        unexp = tf.norm(unexp-tf.nn.embedding_lookup(item_emb_w, self.i) ,ord='euclidean', axis=1) #求和平方根
        print(unexp)#32
        self.unexp = unexp
        print(unexp)#32
        unexp = tf.exp(-1.0*unexp) * unexp *unexp #Unexpected Activation Function
        print(unexp)#32
        #unexp = tf.stop_gradient(unexp)
        print(unexp)#32

        #Relevance (for future exploration)
        relevance = tf.reduce_mean(h_long_emb, axis=1)
        relevance = tf.norm(relevance-tf.nn.embedding_lookup(item_emb_w, self.i) ,ord='euclidean', axis=1)

        #Annoyance/Diversification (for future exploration)
        annoyance = tf.reduce_mean(h_short_emb, axis=1)
        annoyance = tf.norm(annoyance-tf.nn.embedding_lookup(item_emb_w, self.i) ,ord='euclidean', axis=1)

        #Estmation of user preference by combing different components
        self.logits = item_b + concat +user_b+age_b + gender_b+unexp_factor*unexp # [B]exp
        print(self.logits)#32
        self.score = tf.sigmoid(self.logits)
        print(self.score)#32

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,labels=self.y))
        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 1) #梯度下降的慢一点
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0],
                self.hist: uij[1],
                self.i: uij[2],
                self.y: uij[3],
                self.g: uij[4],
                self.a: uij[5],
                self.movie_titles: uij[6],
                #self.genres: uij[6],
                self.lr: lr,
                })
        return loss

    def test(self, sess, uij):
        score, unexp = sess.run([self.score, self.unexp], feed_dict={
                self.u: uij[0],
                self.hist: uij[1],
                self.i: uij[2],
                self.y: uij[3],
                self.g: uij[4],
                self.a: uij[5],
                self.movie_titles: uij[6],
                #self.genres: uij[6],
                })
        return score, uij[3], uij[0], uij[2], unexp,uij[4],uij[5]

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)

    def extract_axis_1(self, data, ind):
        batch_range = tf.range(tf.shape(data)[0])
        indices = tf.stack([batch_range, ind], axis=1)
        res = tf.gather_nd(data, indices)
        return res

    def seq_attention(self, inputs, hidden_size, attention_size):
        """
        Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
        The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
        for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
        Variables notation is also inherited from the article
    
        Args:
            inputs: The Attention inputs.
                Matches outputs of RNN/Bi-RNN layer (not final state):
            attention_size: Linear size of the Attention weights.
            return_alphas: Whether to return attention coefficients variable along with layer's output.
                Used for visualization purpose.
        Returns:
            The Attention output `Tensor`.
            In case of RNN, this will be a `Tensor` shaped:
                `[batch_size, cell.output_size]`.
        """
        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.tile(tf.expand_dims(alphas, -1), [1, 1, hidden_size]), 1, name="attention_embedding")
        return output, alphas

    def unexp_attention(self, querys, keys, keys_id):   #self-attetion MLP
        """
        Same Attention as in the DIN model
        queries:     [Batchsize, 1, embedding_size]
        keys:        [Batchsize, max_seq_len, embedding_size]  max_seq_len is the number of keys(e.g. number of clicked creativeid for each sample)
        keys_id:     [Batchsize, max_seq_len]
        """
        querys = tf.expand_dims(querys, 1)
        keys_length = tf.shape(keys)[1] # padded_dim
        embedding_size = querys.get_shape().as_list()[-1]
        keys = tf.reshape(keys, shape=[-1, keys_length, embedding_size])
        querys = tf.reshape(tf.tile(querys, [1, keys_length, 1]), shape=[-1, keys_length, embedding_size])

        net = tf.concat([keys, keys - querys, querys, keys*querys], axis=-1)
        for units in [32,16]:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        att_wgt = tf.layers.dense(net, units=1, activation=tf.sigmoid)        # shape(batch_size, max_seq_len, 1)
        outputs = tf.reshape(att_wgt, shape=[-1, 1, keys_length], name="weight")  #shape(batch_size, 1, max_seq_len)
        scores = outputs
        scores = scores / (embedding_size ** 0.5)       # scale
        scores = tf.nn.softmax(scores)
        outputs = tf.matmul(scores, keys)    #(batch_size, 1, embedding_size)
        outputs = tf.reduce_sum(outputs, 1, name="unexp_embedding")   #(batch_size, embedding_size)
        return outputs
    def cluster_attention(self, inputs, hidden_size2, attention_size):
        """
        Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
        The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
        for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
        Variables notation is also inherited from the article

        Args:
            inputs: The Attention inputs.
                Matches outputs of RNN/Bi-RNN layer (not final state):
            attention_size: Linear size of the Attention weights.
            return_alphas: Whether to return attention coefficients variable along with layer's output.
                Used for visualization purpose.
        Returns:
            The Attention output `Tensor`.
            In case of RNN, this will be a `Tensor` shaped:
                `[batch_size, cell.output_size]`.
        """
        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size2, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.tile(tf.expand_dims(alphas, -1), [1, 1, hidden_size2]), 1,
                      name="attention_embedding")
        return output, alphas


    def mean_shift(self, input_X, window_radius=0.2): #输入的是item+user embedding
        X1 = tf.expand_dims(tf.transpose(input_X, perm=[0,2,1]), 1) #数组按维度转置

        X2 = tf.expand_dims(input_X, 1)

        C = input_X

        def _mean_shift_step(C):
            C = tf.expand_dims(C, 3)  #插入维度
            Y = tf.reduce_sum(tf.pow((C - X1) / window_radius, 2), axis=2) #沿着维度2做求和降维
            gY = tf.exp(-Y)
            num = tf.reduce_sum(tf.expand_dims(gY, 3) * X2, axis=2)
            denom = tf.reduce_sum(gY, axis=2, keep_dims=True)
            C = num / denom
            return C

        def _mean_shift(i, C, max_diff):
            new_C = _mean_shift_step(C)
            max_diff = tf.reshape(tf.reduce_max(tf.sqrt(tf.reduce_sum(tf.pow(new_C - C, 2), axis=1))), [])
            return i + 1, new_C, max_diff 
        def _cond(i, C, max_diff):
            return max_diff > 1e-5
        n_updates, C , max_diff = tf.while_loop(cond=_cond, body=_mean_shift, loop_vars=(tf.constant(0), C, tf.constant(1e10)))
        return C
        #use this function to embedding movie_categories
    def get_movie_categories_layers(self,movie_categories):
        movie_categories_embed_matrix = tf.Variable(tf.random_uniform([18, 32], -1, 1),
                                                    name="movie_categories_embed_matrix")
        movie_categories_embed_layer = tf.nn.embedding_lookup(movie_categories_embed_matrix, movie_categories,
                                                              name="movie_categories_embed_layer")
        movie_categories_embed_layer = tf.reduce_sum(movie_categories_embed_layer, axis=1, keep_dims=True)
        return movie_categories_embed_layer

    def get_movie_cnn_layer(self,movie_titles):
        window_sizes = {2, 3, 4, 5}
        filter_num = 8
        # 从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
        with tf.name_scope("movie_embedding"):
            movie_title_embed_matrix = tf.Variable(tf.random_uniform([5000, 32], -1, 1),
                                                   name="movie_title_embed_matrix")
            movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, movie_titles,
                                                             name="movie_title_embed_layer")
            movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)

        # 对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
        pool_layer_lst = []
        for window_size in window_sizes:
            with tf.name_scope("movie_txt_conv_maxpool_{}".format(window_size)):
                filter_weights = tf.Variable(tf.truncated_normal([window_size, 32, 1, filter_num], stddev=0.1),
                                             name="filter_weights")
                filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="filter_bias")

                conv_layer = tf.nn.conv2d(movie_title_embed_layer_expand, filter_weights, [1, 1, 1, 1], padding="VALID",
                                          name="conv_layer")
                relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, filter_bias), name="relu_layer")

                maxpool_layer = tf.nn.max_pool(relu_layer, [1, 15 - window_size + 1, 1, 1], [1, 1, 1, 1],
                                               padding="VALID", name="maxpool_layer")
                pool_layer_lst.append(maxpool_layer)

        # Dropout层
        with tf.name_scope("pool_dropout"):
            pool_layer = tf.concat(pool_layer_lst, 3, name="pool_layer")
            max_num = len(window_sizes) * filter_num
            pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num], name="pool_layer_flat")

            dropout_layer = tf.nn.dropout(pool_layer_flat, 1, name="dropout_layer")
        return pool_layer_flat, dropout_layer

