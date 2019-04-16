import json
import pickle
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_dict():
    """加载保存的关系、字符、标签字典"""
    with open("all_50_schemas_me.json","r",encoding="utf8") as inp:
        id2relation,relation2id=json.load(inp)
        id2relation["0"]="N"
        relation2id["N"]=0
        new={}
        for key,val in id2relation.items():
            new[int(key)]=val
        id2relation=new

    with open("all_chars_me.json","r",encoding="utf8") as inp1:
        id2char,char2id=json.load(inp1)
        id2char["0"]="<PAD>"
        id2char["1"]="<UNK>"
        char2id["<PAD>"]=0
        char2id["<UNK>"]=1
        new_char={}
        for key,val in id2char.items():
            new_char[int(key)]=val
        id2char=new_char

    with open("all_ner_label_me.json","r",encoding="utf8") as inp2:
        id2label,label2id=json.load(inp2)
        new_label={}
        for key,val in id2label.items():
            new_label[int(key)]=val
        id2label=new_label

    return id2char,char2id,id2relation,relation2id,id2label,label2id

def embedding(id2char):
    """使用预训练的词向量"""
    np.random.seed(20190403)
    pretrain_embed=[]
    pre_word={}
    with open("./vec.txt","r",encoding="utf8") as inp:
        for line in inp:
            line=line.strip().split()
            word=line[0]
            vec=list(map(float,line[1:]))
            pre_word[word]=vec
            pretrain_embed.append(vec)

    pretrain_embed=np.asarray(pretrain_embed)
    embedding_size=pretrain_embed.shape[1]

    vec_mean=np.mean(pretrain_embed)
    vec_std=np.std(pretrain_embed)
    unknown_embed=np.random.normal(loc=vec_mean,scale=vec_std,size=embedding_size)
    padding_embed=np.random.normal(loc=vec_mean,scale=vec_std,size=embedding_size)

    embedding=[]
    not_num=0
    for key,val in id2char.items():
        if val in pre_word:
            embedding.append(pre_word[val])
        elif key==0:
            embedding.append(padding_embed)
        else:
            embedding.append(unknown_embed)
            not_num+=1

    embedding=np.asarray(embedding)

    with open("embedding.pkl","wb") as outp:
        pickle.dump(embedding,outp)

def mapping_token_to_id(example,char2id):
    """将token映射为ID"""
    token_ids=[]
    for char in example:
        if char in char2id:
            token_ids.append(char2id[char])
        else:
            token_ids.append(char2id["<UNK>"])
    return token_ids

def mapping_label_to_id(example,label2id):
    """将标签映射为ID"""
    BIO_ids=[]
    for label in example:
        BIO_ids.append(label2id[label])
    return BIO_ids

def get_scoreMatrix_head(example,relation2id,max_seq_len=128):
    """head token 分数矩阵"""
    scoreMatrix=np.zeros([max_seq_len,max_seq_len*len(relation2id)])
    for index in range(min(max_seq_len,len(example[0]))):
        relations=example[3][index]
        head=example[4][index]
        label_ids=[]
        for relation in relations:
            label_ids.append(relation2id[relation])
        for k in range(len(relations)):
            col=head[k]*len(relation2id)+label_ids[k]
            scoreMatrix[index][col]=1
    return scoreMatrix

def mapping_to_ids(examples,char2id,label2id,relation2id,max_seq_len=300):
    """将文字和标签序列转换为ids序列，填充序列到等长"""
    doc_ids=[]
    j=0  #仅在本机上设置，服务器上使用全部数据
    for example in tqdm(examples):
        """将序列转换为id序列和获得head score 矩阵"""
        token="".join(example[1])
        token_ids=mapping_token_to_id(example[1],char2id)
        BIO_ids=mapping_label_to_id(example[2],label2id)
        scoreMatrix=get_scoreMatrix_head(example,relation2id,max_seq_len=max_seq_len)

        doc_ids.append((token,token_ids,BIO_ids,scoreMatrix)) #[(文字序列，ids序列，标签ids序列，分数矩阵)]
        j += 1
        if j>=100:
            break
    return doc_ids

def padding(x,max_seq_len=300):
    if len(x)<max_seq_len:
        x.extend([0]*(max_seq_len-len(x)))
    else:
        x=x[:max_seq_len]
    return x

def generator(data,params,config,train=False,shuffle=True):
    #加载图模型中的需要喂数据的参数
    is_train=params["is_train"]
    token=params["token"]
    token_ids=params["token_ids"]
    ner_ids=params["ner_ids"]
    scoreMatrix=params["scoreMatrix"]
    seq_len=params["seq_len"]
    dropout_embedding_keep=params["dropout_embedding_keep"]
    dropout_lstm_keep= params["dropout_lstm_keep"]
    dropout_lstm_output_keep=params["dropout_lstm_output_keep"]
    dropout_fcl_ner_keep=params["dropout_fcl_ner_keep"]
    dropout_fcl_rel_keep=params["dropout_fcl_rel_keep"]

    #打乱数据
    if shuffle:
        data,_,_,_=train_test_split(data,range(len(data)),test_size=0,random_state=20190404)

    batch_num=len(data)//config.batch_size
    all_batch=[]
    batch_token=[]
    batch_token_ids=[]
    batch_ner_ids=[]
    batch_score_matrix=[]
    batch_data_length=[]
    for i in range(len(data)):
        if i%config.batch_size==0 and i>0:
            all_batch.append([batch_token,batch_token_ids,batch_ner_ids,batch_score_matrix,batch_data_length])
            batch_token=[]
            batch_token_ids = []
            batch_ner_ids = []
            batch_score_matrix = []
            batch_data_length = []
        batch_token.append(data[i][0])
        batch_token_ids.append(padding(data[i][1],max_seq_len=config.max_seq_len))
        batch_ner_ids.append(padding(data[i][2],max_seq_len=config.max_seq_len))
        batch_score_matrix.append(data[i][3])
        batch_data_length.append(min(len(data[i][0]),config.max_seq_len))

    if len(data)%config.batch_size!=0:
        batch_num+=1
        all_batch.append([batch_token,batch_token_ids,batch_ner_ids,batch_score_matrix,batch_data_length])

    #调整dropout参数
    if train:
        embedding_keep=config.dropout_embedding_keep
        lstm_keep=config.dropout_lstm_keep
        lstm_output_keep=config.dropout_lstm_output_keep
        fcl_ner_keep=config.dropout_fcl_ner_keep
        fcl_rel_keep=config.dropout_fcl_rel_keep

    else:
        embedding_keep = 1.0
        lstm_keep = 1.0
        lstm_output_keep = 1.0
        fcl_ner_keep = 1.0
        fcl_rel_keep = 1.0
    for iter in range(batch_num):
        yield {is_train:train,
               token:all_batch[iter][0],
               token_ids:all_batch[iter][1],
               ner_ids:all_batch[iter][2],
               scoreMatrix:all_batch[iter][3],
               seq_len:all_batch[iter][4],
               dropout_embedding_keep:embedding_keep,
               dropout_lstm_keep:lstm_keep,
               dropout_lstm_output_keep:lstm_output_keep,
               dropout_fcl_rel_keep:fcl_rel_keep,
               dropout_fcl_ner_keep:fcl_ner_keep}







if __name__ == '__main__':
    a=[ 'O','B-出版社', 'I-出版社', 'B-历史人物', 'I-历史人物', 'B-学科专业', 'I-学科专业', 'B-企业', 'I-企业', 'B-Date',
                    'I-Date', 'B-音乐专辑', 'I-音乐专辑', 'B-机构', 'I-机构', 'B-图书作品', 'I-图书作品', 'B-生物', 'I-生物',
                    'B-Text', 'I-Text', 'B-学校', 'I-学校', 'B-作品', 'I-作品', 'B-目', 'I-目', 'B-人物', 'I-人物',
                    'B-影视作品', 'I-影视作品', 'B-歌曲', 'I-歌曲', 'B-书籍', 'I-书籍', 'B-城市', 'I-城市', 'B-语言', 'I-语言',
                    'B-网络小说', 'I-网络小说', 'B-国家', 'I-国家', 'B-地点', 'I-地点', 'B-气候', 'I-气候', 'B-景点', 'I-景点',
                    'B-网站', 'I-网站', 'B-Number', 'I-Number', 'B-电视综艺', 'I-电视综艺', 'B-行政区', 'I-行政区']

    x=[77, 78, 79, 80, 81, 61, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 11, 94, 64, 95, 96, 97, 98, 99, 100,
     101, 102, 103, 104, 105, 106, 107, 108, 109, 11, 110, 111, 88, 101, 102, 112, 11, 98, 106, 113, 96, 114, 115, 97,
     64, 116, 11, 117, 101, 88, 101, 118, 119, 120, 81, 86, 121, 122, 123, 124, 125, 126, 127, 128, 11, 129, 130, 131,
     132, 133, 134, 135, 136, 8, 129, 118, 108, 137]
    y=[27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 0, 9, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 52, 52, 52, 41,
     42, 42, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    x1=[167, 168, 169, 170, 171, 172, 173, 14, 64, 174, 175, 19, 34, 176, 177, 7, 178, 179, 180, 181, 182, 8, 183, 184, 11,
     185, 186, 34, 8, 187, 153, 188, 189, 190, 168, 191, 192, 29, 193, 152, 194, 195, 196, 34]
    y1=[0, 0, 0, 0, 27, 28, 28, 0, 31, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0]

    id2char, char2id, id2relation, relation2id, id2label, label2id = load_dict()
    get_entity_relations(x1,y1,id2label=id2label,id2char=id2char)


