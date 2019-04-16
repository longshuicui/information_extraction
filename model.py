import tensorflow as tf
import numpy as np
import pandas as pd

class Model(object):
    def __init__(self,config,embed_matrix,sess):
        self.config=config
        self.embed_matrix=embed_matrix
        self.sess=sess

    def run(self):
        is_train=tf.placeholder(tf.bool)
        token=tf.placeholder(tf.string,shape=[None],name="input_token")
        token_ids=tf.placeholder(tf.int32,shape=[None,None],name="input_token_ids")
        ner_ids=tf.placeholder(tf.int32,shape=[None,None],name="ner_task_ids")
        scoreMatrix=tf.placeholder(tf.float32,shape=[None,None,None],name="score_head")
        seq_len=tf.placeholder(tf.int32,[None],name="seq_len")

        dropout_embedding_keep = tf.placeholder(tf.float32, name="dropout_embedding_keep")
        dropout_lstm_keep = tf.placeholder(tf.float32, name="dropout_lstm_keep")
        dropout_lstm_output_keep = tf.placeholder(tf.float32, name="dropout_lstm_output_keep")
        dropout_fcl_ner_keep = tf.placeholder(tf.float32, name="dropout_fcl_ner_keep")
        dropout_fcl_rel_keep = tf.placeholder(tf.float32, name="dropout_fcl_rel_keep")

        embedding_matrix=tf.get_variable(name="embedding",shape=self.embed_matrix.shape,
                                         initializer=tf.constant_initializer(self.embed_matrix))
        input_rnn=tf.nn.embedding_lookup(embedding_matrix,token_ids)
        #mask操作
        mask=tf.sequence_mask(seq_len,maxlen=self.config.max_seq_len,dtype=tf.float32,name="mask")
        mask=tf.expand_dims(mask,-1)
        input_rnn=tf.cast(input_rnn*mask,dtype=tf.float32)

        lossNER,lossRel,predNER,predRel,relScore=self._compute_loss(input_rnn,
                                                                    seq_len=seq_len,
                                                                    ner_ids=ner_ids,
                                                                    scoreMatrix=scoreMatrix,
                                                                    is_train=is_train,
                                                                    dropout_embedding_keep=dropout_embedding_keep,
                                                                    dropout_lstm_keep=dropout_lstm_keep,
                                                                    dropout_lstm_output_keep=dropout_lstm_output_keep,
                                                                    dropout_fcl_ner_keep=dropout_fcl_ner_keep,
                                                                    dropout_fcl_rel_keep=dropout_fcl_rel_keep,
                                                                    mask=mask)

        loss_total=lossRel+lossNER  #整体损失

        rel_true=tf.round(scoreMatrix)  #真实的关系

        #构建一个字典将需要传递数值的参数保存，以便在训练/测试时喂数据
        params={}
        params["is_train"]=is_train
        params["token"]=token
        params["token_ids"]=token_ids
        params["ner_ids"]=ner_ids
        params["scoreMatrix"]=scoreMatrix
        params["seq_len"]=seq_len
        params["dropout_embedding_keep"]=dropout_embedding_keep
        params["dropout_lstm_keep"]=dropout_lstm_keep
        params["dropout_lstm_output_keep"]=dropout_lstm_output_keep
        params["dropout_fcl_ner_keep"]=dropout_fcl_ner_keep
        params["dropout_fcl_rel_keep"]=dropout_fcl_rel_keep

        return loss_total,token,ner_ids,predNER, rel_true,predRel,relScore,params

    def _compute_loss(self,input_rnn,seq_len,ner_ids,scoreMatrix,is_train,dropout_embedding_keep,dropout_lstm_keep,
                      dropout_lstm_output_keep,dropout_fcl_ner_keep,dropout_fcl_rel_keep,mask,reuse=False):
        """计算损失"""
        with tf.variable_scope("loss_conputation",reuse=reuse):
            if self.config.use_dropout:
                input_rnn=tf.nn.dropout(input_rnn,keep_prob=dropout_embedding_keep)

            #BiLSTM底层编码
            # for i in range(self.config.num_lstm_layers):
            #     if self.config.use_dropout and i>0:
            #         input_rnn=tf.nn.dropout(input_rnn,keep_prob=dropout_lstm_keep)
            #     lstm_fw_cell=tf.nn.rnn_cell.LSTMCell(self.config.hidden_size_lstm)
            #     lstm_bw_cell=tf.nn.rnn_cell.LSTMCell(self.config.hidden_size_lstm)
            #
            #     outputs,states=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
            #                                                    lstm_bw_cell,
            #                                                    input_rnn,
            #                                                    sequence_length=seq_len,
            #                                                    dtype=tf.float32,
            #                                                    scope="BiLSTM"+str(i))
            #     input_rnn = tf.concat(outputs, axis=-1)
            #     lstm_output = input_rnn

            input_rnn=tf.keras.layers.CuDNNLSTM(units=self.config.hidden_size_lstm,return_sequences=True)(inputs=input_rnn)
            input_rnn=tf.keras.layers.CuDNNLSTM(units=self.config.hidden_size_lstm,return_sequences=True)(inputs=input_rnn)
            lstm_output=input_rnn

            lstm_output = self._attention(lstm_output,mask)
            if self.config.use_dropout:
                lstm_output=tf.nn.dropout(lstm_output,keep_prob=dropout_lstm_output_keep)

            #计算LSTM发射分数
            nerScores=self._get_ner_score(lstm_output,self.config.num_ner_classes,dropout_fcl_ner_keep)

            #CRF计算NER损失
            log_likelihood,trans_params=tf.contrib.crf.crf_log_likelihood(nerScores,ner_ids,seq_len)
            lossNER=tf.reduce_mean(-log_likelihood)

            #viterbi解码预测NER标签
            predNERS,viterbi_score=tf.contrib.crf.crf_decode(nerScores,trans_params,seq_len)

            #RC输入
            if self.config.label_embedding_size>0:
                labels=tf.cond(is_train,lambda:ner_ids,lambda:predNERS)
                label_matrix=tf.get_variable(name="label_embedding",shape=[self.config.num_ner_classes,
                                                                           self.config.label_embedding_size],dtype=tf.float32)
                label_embedding=tf.nn.embedding_lookup(label_matrix,labels)
                rel_input=tf.concat([lstm_output,label_embedding],axis=2)
            else:
                rel_input=lstm_output

            #关系抽取分数
            relScore=self._get_head_selection_score(rel_input,dropout_keep_in_prob=dropout_fcl_rel_keep)

            #交叉熵计算损失函数
            lossRel=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=relScore,labels=scoreMatrix))
            #预测关系
            probas=tf.nn.sigmoid(relScore)
            predRel=tf.round(probas) #四舍五入

            return lossNER,lossRel,predNERS,predRel,relScore

    def _get_ner_score(self,lstm_output,n_types,dropout_keep_in_prob=1):
        """计算NER发射分数"""
        #两层线性全连接层
        w_1=tf.get_variable("w_1",shape=[self.config.hidden_size_lstm,self.config.hidden_size_n1],dtype=tf.float32)
        b_1 = tf.get_variable("b_1", shape=[self.config.hidden_size_n1], dtype=tf.float32)
        w_2=tf.get_variable("w_2",shape=[self.config.hidden_size_n1,n_types],dtype=tf.float32)
        b_2=tf.get_variable("b_2",shape=[n_types],dtype=tf.float32)

        #第一层
        res=tf.einsum("aij,jk->aik",lstm_output,w_1)+b_1
        res=tf.nn.tanh(res)

        #第二层
        res=tf.einsum("aik,kp->aip",res,w_2)+b_2
        res=tf.nn.tanh(res)

        return res

    def _get_head_selection_score(self,rel_input,dropout_keep_in_prob=1):
        #两两实体进行配对求分数
        v=tf.get_variable("v",shape=[self.config.hidden_size_n1,self.config.num_rel_classes],dtype=tf.float32)
        w_left=tf.get_variable("w_left",shape=[self.config.hidden_size_lstm+self.config.label_embedding_size,self.config.hidden_size_n1],dtype=tf.float32)
        w_right=tf.get_variable("w_right",shape=[self.config.hidden_size_lstm+self.config.label_embedding_size,self.config.hidden_size_n1],dtype=tf.float32)
        b=tf.get_variable("b",shape=[self.config.hidden_size_n1],dtype=tf.float32)

        #矩阵运算求方程 vf(ux+wy+b)
        left=tf.einsum("aij,jk->aik",rel_input,w_left)
        right=tf.einsum("aij,jk->aik",rel_input,w_right)
        output=self._broadcasting(left,right)

        #激活函数
        output=tf.tanh(output+b)
        if self.config.use_dropout:
            output=tf.nn.dropout(output,keep_prob=dropout_keep_in_prob)

        #线性连接
        res=tf.einsum("aijk,kp->aijp",output,v)
        res=tf.reshape(res,shape=[tf.shape(res)[0],tf.shape(res)[1],tf.shape(res)[2]*self.config.num_rel_classes])

        return res

    def _broadcasting(self,left,right):
        left = tf.transpose(left, perm=[1, 0, 2]) #[max_len,batch_size,hidden_size]
        left = tf.expand_dims(left, 3) #[max_len,batch_size,hidden_size,-1]

        right = tf.transpose(right, perm=[0, 2, 1])  #[batch_size,hidden_size,max_len]
        right = tf.expand_dims(right, 0)  #[-1,batch_size,hidden_size,max_len]

        B = left + right #[max_len,batch_size,hidden_size,max_len]
        B = tf.transpose(B, perm=[1, 0, 3, 2])  #[batch_size,max_len,max_len,hidden_size]

        return B

    def _get_train_op(self,loss):
        """训练操作"""
        optimizer=tf.train.AdamOptimizer(self.config.learning_rate)
        if self.config.gradientClipping:
            grad_vars=optimizer.compute_gradients(loss)
            new_grad_vars=[]
            for grad,var in grad_vars:
                if grad==None:
                    grad=tf.zeros_like(var)
                new_grad_vars.append((grad,var))
            clip_grvs=[(tf.clip_by_value(grad,-1.,1.),var) for gard,var in new_grad_vars]
            train_op=optimizer.apply_gradients(clip_grvs)

        else:
            train_op=optimizer.minimize(loss)

        return train_op

    def _attention(self,H,mask):
        """attention"""
        att_w=tf.get_variable(name="att_w",shape=[self.config.max_seq_len,self.config.max_seq_len],dtype=tf.float32)
        H=tf.transpose(H,[0,2,1]) #[batch_size,embedding_size,seq_len]
        weight_att=tf.nn.softmax(att_w)
        H=tf.einsum("aij,jk->aik",H,weight_att)
        H=tf.transpose(H,[0,2,1])  #[batch_size,seq_len.embedding_size]
        H=H*mask

        return H


class Operations():
    """封装模型中的操作，方便后续调用"""
    def __init__(self,train_op,loss,params,token,predNER,actualNER,predRel,actualRel,relScore):
        self.train_op=train_op
        self.loss=loss
        self.params=params
        self.token=token
        self.predNER=predNER
        self.actualNER=actualNER
        self.predRel=predRel
        self.actualRel=actualRel
        self.relScore=relScore









if __name__ == '__main__':
    from main import Config
    import pickle
    config=Config()
    with open("./embedding.pkl","rb") as inp:
        embed_matrix=pickle.load(inp)
    with tf.Session() as sess:
        model=Model(config,embed_matrix,sess)
        model.run()








