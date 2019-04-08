import tensorflow as tf
import pickle
import os
from model import Model,Operations
from utils import *

class Config:
    def __init__(self):
        #adjustment parameter
        self.max_seq_len=300
        self.batch_size=1
        self.learning_rate=0.001
        self.epoches=100
        self.gradientClipping=False

        #network parameter
        self.embedding_size=100
        self.label_embedding_size=0
        self.num_lstm_layers=2
        self.hidden_size_lstm=256
        self.hidden_size_n1=128
        self.num_ner_classes=57
        self.num_rel_classes=50

        #dropout
        self.use_dropout=True
        self.dropout_embedding_keep=0.9
        self.dropout_lstm_keep=0.9
        self.dropout_lstm_output_keep=0.9
        self.dropout_fcl_ner_keep=1.0
        self.dropout_fcl_rel_keep=1.0



if __name__ == '__main__':
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.logging.set_verbosity(tf.logging.INFO)

    config=Config()

    #加载单字字典和关系字典
    id2char,char2id,id2relation,relation2id,id2label,label2id=load_dict()
    print("NER标签数量:%d"%len(id2label))
    print("RC标签数量:%d"%len(id2relation))

    #加载数据
    with open("../data/train.pkl","rb") as inp_train,open("../data/dev.pkl","rb") as inp_dev:
        train_examples=pickle.load(inp_train)
        dev_examples=pickle.load(inp_dev)

    print("训练集样本数量:%d"%len(train_examples))
    print("验证集样本数量:%d"%len(dev_examples))

    train_doc_ids,train_doc_length=mapping_and_padding(train_examples,char2id,label2id,relation2id)
    dev_doc_ids,dev_doc_length=mapping_and_padding(dev_examples,char2id,label2id,relation2id)

    # x=train_doc_ids[5][0]
    # y=train_doc_ids[5][1]
    # z=train_doc_ids[5][2]
    # get_entity_relations(x, y, z, id2label=id2label, id2char=id2char, id2relation=id2relation)
    # exit()

    #加载预训练词向量
    with open("./embedding.pkl","rb") as inp:
        embed_matrix=pickle.load(inp)
    print("词向量维度:",embed_matrix.shape)

    with tf.Session() as sess:
        model=Model(config,embed_matrix,sess)
        #构建图模型
        loss_total, ner_ids, predNER, rel_true, predRel, relScore, params=model.run()
        train_op=model._get_train_op(loss_total)
        operations=Operations(train_op,loss_total,params,predNER,ner_ids,predRel,rel_true,relScore)

        #初始化
        sess.run(tf.global_variables_initializer())

        for iter in range(config.epoches):
            train(train_doc_ids,train_doc_length,config,operations,iter,sess,id2char,id2label)


