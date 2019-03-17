"""
模型超参数定义
数据预处理和生成训练样本
"""
import os
import csv
import pandas as pd
import numpy as np
import ast
import pickle
from collections import Counter


class Config:
    """定义模型超参数"""
    def __init__(self):
        #数据集和词向量地址
        self.train_path=r"multihead_joint_entity_relation_extraction\data\CoNLL04\train.txt"
        self.dev_path=r"multihead_joint_entity_relation_extraction\data\CoNLL04\dev.txt"
        self.test_path=r"multihead_joint_entity_relation_extraction\data\CoNLL04\test.txt"
        self.embedding_path=r"multihead_joint_entity_relation_extraction\data\CoNLL04\vecs.lc.over100freq.txt"
        #训练过程参数
        self.epochs=150
        self.num_tag_types=10
        self.lr=0.001
        self.gradient_clipping=False #是否使用梯度裁剪
        self.use_dropout=False
        self.use_adversarial=False #是否增加对抗机制
        #各层dropout参数
        self.dropou_embedding=0.9
        self.dropout_lstm_input=0.9
        self.dropout_lstm_output=0.9
        self.dropout_fc_ner=1  #实体识别全连接层
        self.dropout_fc_rel=1  #关系抽取全连接层
        #网络超参数
        self.hidden_size_lstm=64 #lstm隐藏层神经元个数
        self.hidden_size_n1=64
        self.hidden_size_n2=32
        self.num_lstm_layers=3
        self.label_embedding_size=0 #如果为0，则不用label_embedding

def load_data(file_path):
    """
    将原始文件处理成DataFrame格式，每一行代表一句话的id,token,tag,rel,head
    :param file_path: 原始文件地址
    :return: 处理好的df格式文件
    """
    col_names = ['token_id', 'token', "BIO", "relation", 'head']
    data=pd.read_csv(file_path,sep="\t",names=col_names,engine="python",quoting=csv.QUOTE_NONE).values
    token_id,token,tag,relation,head=[],[],[],[],[]
    token_ids,tokens,tags,relations,heads=[],[],[],[],[]
    docNr=-1
    for i in range(data.shape[0]):
        if "#doc" in data[i][0] or i==data.shape[0]-1:
            if i==data.shape[0]-1:
                token_id.append(data[i][0])
                token.append(data[i][1])
                tag.append(data[i][2])
                relation.append(data[i][3])
                head.append(data[i][4])
            if docNr!=-1:
                token_ids.append(token_id)
                tokens.append(token)
                tags.append(tag)
                relations.append(relation)
                heads.append(head)
                token_id, token, tag, relation, head = [], [], [], [], []

            docNr+=1
        else:
            token_id.append(data[i][0])
            token.append(data[i][1])
            tag.append(data[i][2])
            relation.append(data[i][3])
            head.append(data[i][4])
    #将每个句子的字段统计
    dict={
        "token_ids":pd.Series([list(map(int,token_id)) for token_id in token_ids]),
        "tokens":pd.Series(tokens),
        "tags":pd.Series(tags),
        "relations":pd.Series([list(map(ast.literal_eval,rel)) for rel in relations]),
        "heads":pd.Series([list(map(ast.literal_eval,head)) for head in heads])
    }

    return pd.DataFrame(dict)

def mapping_to_dict(dataset):
    """
    获取词的集合
    :param dataset: 文档集合
    :return: 正反向字典
    """
    all_elements=[element for line in dataset for element in line]
    counts=Counter(all_elements).most_common(10000)
    element2id={"UNK":0}
    id2element={0:"UNK"}
    for element,_ in counts:
        element2id[element]=len(element2id)
    for key,val in element2id.items():
        id2element[val]=key
    return element2id,id2element

def get_relations_set(dataset):
    relations=[]
    for line in dataset:
        for token in line:
            for relation in token:
                relations.append(relation)
    relations=sorted(list(set(relations)))
    return relations

def get_score_matrix(relations,set_rel,heads):
    """获得关系分数矩阵"""
    joint_ids=[]
    for i in range(len(relations)):
        doc_joint_ids=[]
        for j in range(len(relations[i])):
            relation_ids=[]
            for rel in relations[i][j]:
                relation_ids.append(set_rel.index(rel))
            score_matrix=[]
            for index in range(len(relation_ids)):
                score_matrix.append(heads[i][j][index]*len(set_rel)+relation_ids[index])
            doc_joint_ids.append(score_matrix)
        joint_ids.append(doc_joint_ids)

    score_matrix=[]
    for i in range(len(relations)):
        scoringMatrix = np.zeros([len(relations[i]), len(relations[i]) * len(set_rel)])
        for j in range(len(relations[i])):
            tokenHead=joint_ids[i][j]
            for head in tokenHead:
                scoringMatrix[j,head]=1
        score_matrix.append(scoringMatrix)
    return score_matrix

def pretrain_embedding(vec_path=None,word2id=None):
    embedding=dict()
    with open(vec_path,"r",encoding="utf8",errors="ignore") as inp:
        for line in inp:
            line=line.strip().split()
            embedding[line[0]]=list(map(float,line[1:]))
    embed_matrix=[]
    for word in word2id:
        if word in embedding:
            embed_matrix.append(embedding[word])
        else:
            embed_matrix.append(np.random.random(50))
    embed_matrix=np.asarray(embed_matrix)
    print("预训练词向量维度",embed_matrix.shape)
    with open("embed_matrix.pkl","wb") as outp:
        pickle.dump(embed_matrix,outp)

    return embed_matrix

def batch_generator(data,batch_size=16,shuffle=True):
    num_doc=len(data)
    if shuffle:
        shuffle_ids=np.random.permutation(num_doc)
        data=data.iloc[shuffle_ids,:]
    num_batch=num_doc//batch_size  #56个batch
    batch_tokens_ids,batch_tags_ids,batch_relations,batch_heads=[],[],[],[]











if __name__ == '__main__':
    config=Config()

    print("Start load data...\n")
    data_train=load_data(config.train_path)  #(910, 5)
    data_test=load_data(config.test_path)  #(288, 5)
    data_dev=load_data(config.dev_path)  #(243, 5)
    # batch_generator(data_train)
    # exit()

    tokens=list(data_train.tokens)+list(data_test.tokens)+list(data_dev.tokens)
    tags=list(data_train.tags)+list(data_dev.tags)+list(data_test.tags)
    relations=list(data_train.relations)+list(data_dev.relations)+list(data_test.relations)

    print("Start mapping to id...\n")
    word2id,id2word=mapping_to_dict(tokens)
    tag2id,id2tag=mapping_to_dict(tags)
    relations_set=get_relations_set(relations)
    print(relations_set)
    score_matrix=get_score_matrix(list(data_train.relations),relations_set,list(data_train.heads))


    print("Load pretrain embedding...\n")
    if os.path.exists("embed_matrix.pkl"):
        with open("embed_matrix.pkl","rb") as inp:
            embed_matrix=pickle.load(inp)
    else:
        embed_matrix=pretrain_embedding(config.embedding_path,word2id=word2id)

