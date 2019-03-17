import tensorflow as tf

class Model:
    def __init__(self,config,emb_matrix,sess):
        self.config=config
        self.emb_matrix=emb_matrix
        self.sess=sess

        self.embedding_ids=tf.placeholder(dtype=tf.int32,shape=[None,None],name="embedding_ids")  #输入词的ids化序列
        self.entity_tags_ids=tf.placeholder(dtype=tf.int32,shape=[None,None],name="entity_tags_ids") #输入词的标记的ids序列
        self.score_matrix_true=tf.placeholder(dtype=tf.float32,shape=[None,None,None],name="score_matrix_true") #真实矩阵分数
        self.seq_length=tf.placeholder(dtype=tf.int32,shape=[None],name="seq_length")  #输入的真正的序列长度
        self.dropout_embedding_keep=tf.placeholder(dtype=tf.float32,name="dropout_embedding_keep")
        self.dropout_lstm_keep=tf.placeholder(tf.float32,name="dropout_lstm_keep")
        self.dropout_lstm_output_keep=tf.placeholder(tf.float32,name="output_keep")
        self.dropout_ner_keep=tf.placeholder(tf.float32,name="ner_keep")
        self.dropout_rel_keep=tf.placeholder(tf.float32,name="rel_keep")

        with tf.name_scope("embedding_lookup"):
            word_embedding=tf.get_variable(name="word_embedding",shape=self.emb_matrix.shape,dtype=tf.float32,initializer=tf.random_normal_initializer)
            if self.config.use_pretrain_emb:
                word_embedding=word_embedding.assign(tf.cast(emb_matrix,tf.float32),name="emebdding")
            embedding_input=tf.nn.embedding_lookup(word_embedding,self.embedding_ids)
        loss_ner,loss_rel,self.pred_seq,self.pred_rel,rel_score=self.computer_loss(embedding_input,is_train=self.config.is_train)

        #计算总体损失
        self.loss_all=tf.reduce_mean(loss_ner+loss_rel)
        """
        增加对抗
        """
        self.rel_true=tf.round(self.score_matrix_true) #真实关系

        with tf.name_scope("train_Op"):
            optimizer=tf.train.AdamOptimizer(learning_rate=self.config.lr)
            if self.config.gradient_clipping:
                #进行梯度裁剪
                grad_var=optimizer.compute_gradients(self.loss_all)
                new_grad = []
                for grad,var in grad_var:
                    if grad==None:
                        grad=tf.zeros_like(var)
                    new_grad.append([grad,var])
                if len(new_grad)!=len(grad_var):
                    tf.logging.info("gradient error")
                clipped_gvs=[(tf.clip_by_value(grad,-1.,1.),var) for grad,var in new_grad]
                self.train_op=optimizer.apply_gradients(clipped_gvs)
            else:
                self.train_op=optimizer.minimize(self.loss_all)




    def computer_loss(self,input_rnn,is_train=True,reuse=False):
        """计算实体识别和关系抽取联合损失"""
        with tf.variable_scope("loss_computation",reuse=reuse):
            if self.config.use_dropout:
                input_rnn=tf.nn.dropout(input_rnn,keep_prob=self.dropout_embedding_keep)
                tf.logging.info("the shape of input_rnn:",input_rnn.shape)
            for i in range(self.config.num_lstm_layers):
                if self.config.use_dropout and i>0:
                    input_rnn=tf.nn.dropout(input_rnn,keep_prob=self.dropout_lstm_keep)
                lstm_fw=tf.nn.rnn_cell.BasicLSTMCell(self.config.num_hidden_lstm)
                lstm_bw=tf.nn.rnn_cell.BasicLSTMCell(self.config.num_hidden_lstm)
                outputs,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw,lstm_bw,input_rnn,
                                                          sequence_length=self.seq_length,dtype=tf.float32)
                input_rnn=tf.concat(outputs,axis=2)
                lstm_output=input_rnn

            if self.config.use_dropout:
                lstm_output=tf.nn.dropout(lstm_output,keep_prob=self.dropout_lstm_output_keep)
            tf.logging.info("the shape of bilstm_output:",lstm_output.get_shape())
            #计算发射分数
            w_1=tf.get_variable(name="w1",shape=[self.config.num_hidden_lstm*2,self.config.hidden_size])
            b_1=tf.get_variable(name="b1",shape=[self.config.hidden_size])
            w_2=tf.get_variable(name="w2",shape=[self.config.hidden_size,self.config.num_tag_types])
            b_2=tf.get_variable(name="b2",shape=[self.config.num_tag_types])
            #先对lstm层的输出做一个线性映射
            out=tf.einsum("aij,jk->aik",lstm_output,w_1)+b_1
            out=tf.tanh(out)
            emit_score=tf.einsum("aik,kp->aip",out,w_2)+b_2

            #用crf计算转移分数
            log_likelihood,trans_param=tf.contrib.crf.crf_log_likelihood(emit_score,self.entity_tags_ids,self.seq_length)
            pred_seq,viterbi_score=tf.contrib.crf.crf_decode(emit_score,trans_param,self.seq_length)

            loss_ner=-log_likelihood

            if self.config.label_embedding_size>0:
                label_matrix=tf.get_variable(name="label_embed",shape=[self.config.num_tag_types,self.config.label_embedding_size])
                if is_train:
                    tags=self.entity_tags_ids
                else:
                    tags=pred_seq
                label_embeddings=tf.nn.embedding_lookup(label_matrix,tags)

                rel_input=tf.concat([lstm_output,label_embeddings],axis=2) #作为关系抽取层的输入
            else:
                rel_input=lstm_output

            #计算关系抽取层的损失
            w_3=tf.get_variable(name="w3",shape=[self.config.num_hidden_lstm*2+self.config.label_embedding_size,self.config.hidden_size])  #[2*h_lstm+label_emb,hidden_size]从前到后的权重
            w_4=tf.get_variable(name="w4",shape=[self.config.num_hidden_lstm*2+self.config.label_embedding_size,self.config.hidden_size])  #[2*h_lstm+label_emb,hidden_size]从后到前的权重
            w_5=tf.get_variable(name="w5",shape=[self.config.hidden_size,self.config.num_rel_types])  #[hidden_size,rel_num] 向关系层映射是哪有关系
            b_3=tf.get_variable(name="b3",shape=[self.config.hidden_size])  #偏置项

            left=tf.einsum("aij,jk->aik",rel_input,w_3) #从前到后的得分 [batch_size,max_len,hiddin_size]
            right=tf.einsum("aij,jk->aik",rel_input,w_4) #从后到前的得分[batch_size,max_len,hiddin_size]
            tf.logging.info("the shape of left or right:",left.get_shape())

            left=tf.transpose(left,[1,0,2])
            left=tf.expand_dims(left,axis=3) #[max_len,batch_size,hidden_size,---]
            right=tf.transpose(right,[0,2,1])
            right=tf.expand_dims(right,axis=0) #[---,batch_size,hidden_size,max_len]
            out_score=left+right  #将从前到后和从后到前的分数相加
            out_score=tf.transpose(out_score,[1,0,3,2])+b_3 #获得每个词与其他词的关系分数
            out_score=tf.tanh(out_score)
            #再做一次线性变化，映射到关系类别空间
            out_score=tf.einsum("aijk,kp->aijp",out_score,w_5)
            shape=out_score.shape
            rel_score=tf.reshape(out_score,shape=[shape[0],shape[1],shape[2]*shape[3]])

            loss_rel=tf.nn.softmax_cross_entropy_with_logits(logits=rel_score,labels=self.score_matrix_true)
            # 分数转换为概率
            probas = tf.nn.sigmoid(rel_scores)
            # 获取预测的关系
            pred_rel = tf.round(probas)
            return loss_ner,loss_rel,pred_seq,pred_rel,rel_score

















if __name__ == '__main__':
    model=Model("","","")
