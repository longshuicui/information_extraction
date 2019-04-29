import json
import numpy as np
import time
import os
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from attention import SelfAttention
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#加载关系/字映射字典
relation2id,id2relation=json.load(open("./all_50_schemas.json",encoding="utf8"))
char2id,id2char=json.load(open("./all_char_dict.json",encoding="utf8"))
print("关系个数，%d"%len(relation2id))
print("字符个数，%d"%len(char2id))

#加载训练验证测试数据
train_data=json.load(open("train_data.json",encoding="utf8"))
dev_data=json.load(open("dev_data.json",encoding="utf8"))
test_data=json.load(open("test_data.json",encoding="utf8"))
print("训练样本数量，%d"%len(train_data))
print("验证样本数量，%d"%len(dev_data))
print("测试集数量，%d"%len(test_data))

#加载预训练词向量
vec=np.load("word2vec.npy")
print("词向量维度，",vec.shape)

#模型超参
epoches=10
embedding_size=100
n_type=len(relation2id)
char_size=len(char2id)+2

def process(examples,max_seq_len):
    token_ids=[]
    score_matrixs=[]
    for example in examples:
        score_matrix = np.zeros([max_seq_len,max_seq_len,n_type],dtype=int)
        text=example["text"]
        l=len(text)
        spo_list=example["spo_list"]
        for spo in spo_list:
            subject=spo["subject"]
            object=spo["object"]
            relation_id=relation2id[spo["predicate"]]

            sub_start=text.find(subject)  #主体起始位置
            sub_end=sub_start+len(subject)-1  #主体终止位置
            obj_start=text.find(object)  #客体起始位置
            obj_end=obj_start+len(object)-1  #客体终止位置
            score_matrix[sub_start:sub_end+1,obj_start:obj_end+1,relation_id]=1
        score_matrixs.append(score_matrix)
        #token_序列
        token_id=[]
        for char in text:
            if char in char2id:
                token_id.append(char2id[char])
            else:
                token_id.append(1) #UNK
        if len(token_id)<max_seq_len:
            token_id.extend([0]*(max_seq_len-len(token_id)))  #PAD
        else:
            token_id=token_id[:max_seq_len]
        token_ids.append(token_id)
    return token_ids,score_matrixs

def generator(data,batch_size=32):
    data,_,_,_=train_test_split(data,range(len(data)),test_size=0,random_state=20190422)  #shuffle

    batch_num=len(data)//batch_size
    if len(data)%batch_size!=0:
        batch_num+=1
    batch_data=[]
    all_batch=[]
    for i in range(len(data)):
        if i%batch_size==0 and i>0:
            all_batch.append(batch_data)
            batch_data=[]
        batch_data.append(data[i])
        # if i>63:
        #     break
    all_batch.append(batch_data)
    for chunk in all_batch:
        max_seq_len=max([len(d["text"]) for d in chunk])
        token_ids,score_matrixs=process(chunk, max_seq_len=max_seq_len)
        yield token_ids,score_matrixs,max_seq_len

def get_spo_list(token,score_matrix):
    """
    [{'subject': '周华健', 'predicate': '妻子', 'object': '康粹兰'}, {'subject': '康粹兰', 'predicate': '丈夫', 'object': '周华健'}]
    """
    total_rel = []
    for i in range(len(token)):  #寻找每个字符之间对应的关系，关系相同的组成一个实体
        per_char_rel = []
        rel_set = set()
        sub_relation = score_matrix[i]  #主体对应的每个字符
        for j in range(len(token)):
            link=np.argmax(sub_relation[j])
            relation=id2relation[str(link)]
            rel_set.add(relation)
            per_char_rel.append((token[i], relation, token[j]))  #主体字符，关系，宾语字符
        # 相同的实体
        objects=[]
        object_=[]
        for k in range(len(per_char_rel)-1):
            if per_char_rel[k][1]!="N":
                object_.append(per_char_rel[k][2])
            if len(object_)!=0 and per_char_rel[k+1][1]!=per_char_rel[k][1]:
                object_="".join(object_)
                objects.append([object_,per_char_rel[k][1]])
                object_=[]
        for object in objects:
            total_rel.append([token[i],object])

    #将subject拿出来，这里失去了边界信息，对于多对一的问题无法解决，
    #如果先提取object，那么一对多的问题无法解决，解码方式的差别
    objects = set()
    for item in total_rel:
        objects.add(item[1][0])

    spo_list = []
    for object in objects:
        subject = []
        predicate = []
        if object == "":
            continue
        for item in total_rel:
            if item[1][0] == object:
                subject.append(item[0])
                predicate.append(item[1][1])
        # 这里相同的obj，假设仅存在于一个关系
        spo = {"subject": "".join(subject), "predicate": predicate[0], "object": object}
        spo_list.append(spo)

    return spo_list

def broadcasting(left,right):
    left = tf.transpose(left, perm=[1, 0, 2]) #[max_len,batch_size,hidden_size]
    left = tf.expand_dims(left, 3) #[max_len,batch_size,hidden_size,-1]

    right = tf.transpose(right, perm=[0, 2, 1])  #[batch_size,hidden_size,max_len]
    right = tf.expand_dims(right, 0)  #[-1,batch_size,hidden_size,max_len]

    B = left + right #[max_len,batch_size,hidden_size,max_len]
    B = tf.transpose(B, perm=[1, 0, 3, 2])  #[batch_size,max_len,max_len,hidden_size]
    return B

def create_model():
    token_ids=tf.placeholder(tf.int32,shape=(None,None),name="token_ids")
    score_matrix=tf.placeholder(tf.float32,shape=(None,None,None,n_type),name="score_matrix")
    max_seq_len=tf.placeholder(tf.int32,name="max_seq_len")

    mask=tf.cast(tf.greater(tf.expand_dims(token_ids,2),0),dtype=tf.float32)
    embedding=tf.get_variable(name="embedding",initializer=tf.cast(vec,tf.float32))
    # embedding=tf.get_variable(name="embedding",dtype=tf.float32,shape=[char_size,100],initializer=tf.random_normal_initializer)
    inp_embed=tf.nn.embedding_lookup(embedding,token_ids)
    inp=inp_embed*mask

    inp=Bidirectional(LSTM(units=256,return_sequences=True))(inp)  #GPU上使用CuDNNLSTM加速  CPU上使用LSTM
    inp=Bidirectional(LSTM(units=256,return_sequences=True))(inp)

    #ATT+BN
    self_attention=SelfAttention(d_model=512)
    lstm_output=self_attention(token_ids,inp)
    lstm_output=tf.nn.batch_normalization(lstm_output,mean=0.0,variance=3.0,offset=None,scale=1.0,variance_epsilon=0.01,name="BN1")

    # lstm_output=inp
    left=Conv1D(filters=64,kernel_size=3,activation="relu",padding="same")(lstm_output)
    right=Conv1D(filters=64,kernel_size=3,activation="relu",padding="same")(lstm_output)
    out=broadcasting(left,right)
    out = tf.nn.batch_normalization(out, mean=0.0, variance=3.0, offset=None, scale=1.0, variance_epsilon=0.01,name="BN2")

    pred_score=Dense(units=50,activation="softmax")(out)

    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_score,labels=score_matrix))

    optimizer=tf.train.AdamOptimizer(learning_rate=0.001)
    train_op=optimizer.minimize(loss)
    return token_ids,score_matrix,max_seq_len,pred_score,loss,train_op

def main():
    with tf.Session() as sess:
        token_ids,score_matrix,max_seq_len,pred_score,loss,train_op=create_model()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        best_f=0.4
        for iter in range(epoches):
            print("**********Train--%d**********"%iter)
            num,train_loss=0,0
            start=time.time()
            for batch_x,batch_y,batch_z in generator(train_data):
                _,pred,l=sess.run([train_op,pred_score,loss],feed_dict={token_ids:batch_x,score_matrix:batch_y,max_seq_len:batch_z})
                num+=1
                train_loss+=l
            end=time.time()
            print("train_loss:{:.3f},  spend time: {:.3f}s".format(train_loss,end-start))

            #验证过程
            A,B,C=1e-10,1e-10,1e-10
            start=time.time()
            for data in tqdm(dev_data):
                token_,true_matrix=process([data],max_seq_len=len(data["text"]))
                pred_matrix=sess.run(pred_score,feed_dict={token_ids:np.array(token_),
                                                           max_seq_len:len(data["text"])})

                pred_spo_list=get_spo_list(data["text"],pred_matrix[0])
                true_spo_list=get_spo_list(data["text"],true_matrix[0])

                cross=[i for i in true_spo_list if i in pred_spo_list]
                A+=len(cross)
                B+=len(pred_spo_list)
                C+=len(true_spo_list)

            end=time.time()
            p=A/B
            r=A/C
            f=2*A/(B+C)
            print("dev--spend time:{:.3f}--precision={:.3f},  recall={:.3f},  f1={:.3f}".format(end-start,p,r,f))

            if f>best_f:
                best_f=f
                path=saver.save(sess,"output/model.ckpt")
                print("当前模型已保存，路径地址：",path)
            print()

        # saver.restore(sess,tf.train.latest_checkpoint("./output"))
        print("开始测试集输出")
        f=open("res.json","w",encoding="utf8")
        for test in tqdm(test_data):
            text=test["text"]
            inp=[char2id[char] for char in text]
            test_matrix=sess.run(pred_score,feed_dict={token_ids:[inp],max_seq_len:len(text)})
            pred_spo=get_spo_list(text,test_matrix[0])
            s=json.dumps({"text":text,"spo_list":pred_spo},ensure_ascii=False)
            f.write(s+"\n")
        f.close()














if __name__ == '__main__':
    # data=train_data[111111]
    # print(data["spo_list"])
    # token_ids,score_matrix=process([data],len(data["text"]))
    # score_matrix=score_matrix[0]
    # token=data["text"]
    # print(token)
    # spo_list=get_spo_list(token,score_matrix)
    # print(spo_list)

    main()








