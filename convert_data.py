import json
from tqdm import tqdm
import jieba
from jieba import posseg as psg
import pandas as pd
import numpy as np
import pickle

def pos_tag(text):
    jieba.load_userdict("./add_word.txt")
    res=psg.cut(text)
    pos=[]
    for i,j in res:
        pos.append(i+"/"+j)
    return " ".join(pos)

pos_none=0   #50个没有分词的
k=0
examples=[]
with open("./train_data.json","r",encoding="utf8") as inp:
    for line in inp:
        line=json.loads(line)  #['postag', 'text', 'spo_list']
        postag=line["postag"] #词性标注
        text=line["text"].lower()  #未处理文本
        spo_list=line["spo_list"]  #关系集合

        index=[i for i in range(len(text))] #字的位置
        char_list=list(text)
        pos_list=[]
        relations=[]
        obj_position=[]

        spo_id=0
        for spo in spo_list:
            try:
                sub_start=text.index(spo["subject"])    #主语在文本中的开始下标
                sub_end=sub_start+len(spo["subject"])-1 #主语在文本中的结束下标
                obj_start=text.index(spo["object"])     #宾语在文本中的开始下标
                obj_end=obj_start+len(spo["object"])-1  #宾语在文本中的结束下标
            except ValueError:
                print(text)
                continue

            for i in range(len(text)):
                if i>=sub_start and i<=sub_end:
                    if spo_id==0:
                        if i==sub_start:
                            pos_list.append("B-"+spo["subject_type"])
                            relations.append(["N"])
                            obj_position.append([i])
                        elif i==sub_end:
                            pos_list.append("I-" + spo["subject_type"])
                            relations.append([spo["predicate"]])
                            obj_position.append([obj_end])
                        else:
                            pos_list.append("I-"+spo["subject_type"])
                            relations.append(["N"])
                            obj_position.append([i])
                    else: #如果存在就需要更改
                        if i == sub_start:
                            pos_list[i]=("B-" + spo["subject_type"])
                            relations[i]=["N"]
                            obj_position[i]=[i]
                        elif i == sub_end:
                            pos_list[i]="I-" + spo["subject_type"]
                            relations[i].append(spo["predicate"])
                            obj_position[i].append(obj_end)
                        else:
                            pos_list[i]="I-" + spo["subject_type"]
                            relations[i]=["N"]
                            obj_position[i]=[i]

                elif i>=obj_start and i<=obj_end:
                    if spo_id==0:
                        if i==obj_start:
                            pos_list.append("B-"+spo["object_type"])
                        else:
                            pos_list.append("I-" + spo["object_type"])

                        relations.append(["N"])
                        obj_position.append([i])
                    else:
                        if i == obj_start:
                            pos_list[i]="B-" + spo["object_type"]
                        else:
                            pos_list[i]="I-" + spo["object_type"]

                        relations[i]=["N"]
                        obj_position[i]=[i]


                else:
                    if spo_id==0:
                        pos_list.append("O")
                        relations.append(["N"])
                        obj_position.append([i])
            spo_id+=1


        # df=pd.DataFrame({"index":pd.Series(index),
        #                  "word":pd.Series(char_list),
        #                  "pos":pd.Series(pos_list),
        #                  "relation":pd.Series(relations),
        #                  "obj_pos":pd.Series(obj_position)})

        # for iter in range(len(text)):
        #     try:
        #         s=str(index[iter])+"\t"+char_list[iter]+"\t"+pos_list[iter]+"\t"+str(relations[iter])+"\t"+str(obj_position[iter])
        #         outp.write(s+"\n")
        #     except IndexError:
        #         break

        # outp.write("\n")
        if len(relations)>0:
            examples.append((index,char_list,pos_list,relations,obj_position))
print("****************************************")
print(examples[47])

with open("./train.pkl","wb") as outp:
    pickle.dump(examples,outp)

print("数据已经保存")








