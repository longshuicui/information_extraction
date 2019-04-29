
#************word2vec 训练词向量********************
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import json
import time

train_data=[d["text"] for d in json.load(open('train_data.json',encoding="utf8"))]
dev_data=[d["text"] for d in json.load(open("dev_data.json",encoding="utf8"))]
test_data=[d["text"] for d in json.load(open("test_data.json",encoding="utf8"))]
total_data=train_data+dev_data+test_data
corpus=[list(text) for text in total_data]

start=time.time()
model=Word2Vec(sentences=corpus,size=100,max_vocab_size=8421,min_count=2,iter=5)
model.wv.save_word2vec_format("./word2vec.txt")
end=time.time()
print("Done,spend:%.2f"%(end-start))


import numpy as np
id2char=json.load(open("all_char_dict.json",encoding="utf8"))[0]

pretrain={}
with open("word2vec.txt","r",encoding="utf8") as inp:
    for line in inp:
        line=line.strip().split()
        pretrain[line[0]]=list(map(float,line[1:]))
pretrain["UNK"]=np.random.random(100)
pretrain["PAD"]=np.random.random(100)
word2vec=[]
word2vec.append(pretrain["PAD"])
word2vec.append(pretrain["UNK"])

for i,char in id2char.items():
    if char in pretrain:
        word2vec.append(pretrain[char])
    else:
        word2vec.append(pretrain["UNK"])

word2vec=np.asarray(word2vec)
np.save("word2vec.npy",word2vec)
print(word2vec.shape)

