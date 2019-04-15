import tensorflow as tf
import pickle
import os
import json
from tqdm import tqdm
from model import Model,Operations
from utils import *
from process import *

class Config:
    def __init__(self):
        #adjustment parameter
        self.max_seq_len=300
        self.batch_size=32
        self.learning_rate=0.001
        self.epoches=1
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
    tf.logging.info("NER标签数量:%d"%len(id2label))
    tf.logging.info("RC标签数量:%d"%len(id2relation))

    # 加载预训练词向量
    with open("./embedding.pkl", "rb") as inp:
        embed_matrix = pickle.load(inp)
    tf.logging.info("词向量维度:%s"%str(embed_matrix.shape))

    #加载数据
    with open("../data/train.pkl","rb") as inp_train,open("../data/dev.pkl","rb") as inp_dev:
        train_examples=pickle.load(inp_train)
        dev_examples=pickle.load(inp_dev)

    train_doc_ids=mapping_to_ids(train_examples,char2id,label2id,relation2id)
    dev_doc_ids=mapping_to_ids(dev_examples,char2id,label2id,relation2id)

    tf.logging.info("训练集样本数量:%d" % len(train_doc_ids))
    tf.logging.info("验证集样本数量:%d" % len(dev_doc_ids))
    tf.logging.info("the example of train")
    tf.logging.info("token：%s"%str(train_doc_ids[0][0]))
    tf.logging.info("token_id：%s"%str(train_doc_ids[0][1]))
    tf.logging.info("ner_id：%s"%str(train_doc_ids[0][2]))

    with tf.Session() as sess:
        model=Model(config,embed_matrix,sess)
        #构建图模型
        loss_total, token, ner_ids, predNER, rel_true, predRel, relScore, params=model.run()
        train_op=model._get_train_op(loss_total)
        operations=Operations(train_op,loss_total,params,token,predNER,ner_ids,predRel,rel_true,relScore)

        #初始化
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for iter in range(config.epoches):
            train(train_doc_ids,config,operations,iter,sess,id2label=id2label,id2relation=id2relation)
            dev(dev_doc_ids,config,operations,sess,id2relation=id2relation,id2label=id2label)

            if (iter+1)%5==0:
                path=saver.save(sess,"output/model.ckpt")
                print("模型已经保存，路径地址：",path)

        print("开始进行测试集输出.......")

        outp=open("./res.json","w",encoding="utf8")
        test_datas=json.load(open("../data/test_data_me.json",encoding="utf8"))
        for data in tqdm(test_datas):
            spo_list=test(data["text"], config, operations, sess, params,id2relation=id2relation, id2label=id2label, char2id=char2id)

            single = {"text": data["text"],
                      "spo_list": spo_list}
            s=json.dumps(single,ensure_ascii=False)
            outp.write(s+"\n")

        outp.close()
