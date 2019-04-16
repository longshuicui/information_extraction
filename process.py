import json
import pickle
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score,recall_score,f1_score
from utils import *

def get_spo_list(tokens,rel_score_matrix,ner_ids,id2relation=None,id2label=None):
    """获取实体之间的关系(subject predicate object)"""
    relations=[]
    for ids in range(len(tokens)):
        per_relation=[]
        score=rel_score_matrix[ids]
        for j in range(len(tokens)):
            cls=score[j*len(id2relation):(j+1)*len(id2relation)]
            if cls.sum()>0.:
                index=cls.argmax(axis=0)
                rel=id2relation[index]
                per_relation.append((ids,rel,j))
        relations.append(per_relation)

    spo_list=[]
    for rel in relations:
        for tuple in rel:
            if tuple[1]=="N":
                continue
            sub_end=tuple[0]
            predicate=tuple[1]
            obj_end=tuple[2]

            #寻找subject
            if id2label[ner_ids[sub_end]]=="O":
                continue
            subject_type = id2label[ner_ids[sub_end]].split("-")[1]  #主语的类型
            subject=""
            for i in range(sub_end,-1,-1):
                if id2label[ner_ids[i]]=="B-"+subject_type:
                    subject=tokens[i:sub_end+1]
                    break
            #寻找object
            if id2label[ner_ids[obj_end]]=="O":
                continue
            object_type=id2label[ner_ids[obj_end]].split("-")[1]  #宾语类型
            object=""
            for i in range(obj_end,-1,-1):
                if id2label[ner_ids[i]]=="B-"+object_type:
                    object=tokens[i:obj_end+1]
                    break
            if subject=="" or object=="":
                continue
            spo={"object_type":object_type,"predicate":predicate,"object":object,"subject_type":subject_type,"subject":subject}
            spo_list.append(spo)

    return spo_list


def train(train_data,config,operations,iter,sess,id2label=None,id2relation=None):
    print("#####**Train iter %d**#####" % iter)
    loss,j=0,0
    start=time.time()
    for x_train in generator(train_data,operations.params,config,train=True):
        _,batch_loss,predNER,actualNER,predRel,actualRel,tokens=sess.run([operations.train_op,
                                                                          operations.loss,
                                                                          operations.predNER,
                                                                          operations.actualNER,
                                                                          operations.predRel,
                                                                          operations.actualRel,
                                                                          operations.token],feed_dict=x_train)

        loss+=batch_loss
        j += 1
    end = time.time()
    print("loss:%.4f"%(loss/j))
    print("Current step spend time:%.2f"%(end-start))
    print()


def dev(dev_data,config,operations,sess,id2relation=None,id2label=None):
    print("-------Evaluate----------")
    y_true,y_pred=[],[]
    a=1e-10
    for x_dev in generator(dev_data,operations.params,config,train=False):
        batch_loss, predNER, actualNER, predRel, actualRel,tokens = sess.run([operations.loss,
                                                                              operations.predNER,
                                                                              operations.actualNER,
                                                                              operations.predRel,
                                                                              operations.actualRel,
                                                                              operations.token],
                                                                              feed_dict=x_dev)

        for i in range(len(tokens)):
            pred_spo_list = get_spo_list(tokens[i], predRel[i], predNER[i], id2label=id2label, id2relation=id2relation)
            true_spo_list = get_spo_list(tokens[i], actualRel[i],actualNER[i],id2label=id2label,id2relation=id2relation)
            if pred_spo_list==true_spo_list:
                a+=1
            y_true.append(true_spo_list)
            y_pred.append(pred_spo_list)

    p=a/len(y_pred)
    r=a/len(y_true)
    f=2*p*r/(p+r)

    print("Current dev data,precision={:.4f},recall={:.4f},f1={:.4f}".format(p,r,f))
    print()


def test(test_data,config,operations,sess,params,id2relation=None,id2label=None,char2id=None):
    feed_dict={params["is_train"]:False,
               params["token"]:[test_data],
               params["token_ids"]:[padding(mapping_token_to_id(test_data,char2id=char2id))],
               params["seq_len"]:[len(test_data)],
               params["dropout_embedding_keep"]:1,
               params["dropout_lstm_keep"]:1,
               params["dropout_lstm_output_keep"]:1,
               params["dropout_fcl_ner_keep"]:1,
               params["dropout_fcl_rel_keep"]:1}
    predNER,predRel,tokens=sess.run([operations.predNER,operations.predRel,operations.token],feed_dict=feed_dict)

    spo_list=get_spo_list(tokens[0],predRel[0],predNER[0],id2relation=id2relation,id2label=id2label)

    return spo_list



if __name__ == '__main__':
    with open("../data/train.pkl","rb") as inp:
        train_data=pickle.load(inp)
    with open("../data/dev.pkl","rb") as inp:
        dev_data=pickle.load(inp)

    length=[]
    for line in train_data:
        length.append(len(line[0]))
    for line in dev_data:
        length.append(len(line[0]))

    print("最大长度",max(length))
    print("最小长度",min(length))
    print("平均长度",sum(length)/len(length))

    for i in range(len(length)):
        if length[i]>128:
            length[i]=1
        else:
            length[i]=0

    print(sum(length)/len(length))



