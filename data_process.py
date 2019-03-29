import json
import pandas as pd
import numpy as np


def entity_type_extract():
    """提取实体类型和关系类型"""
    subject_type=[] #主体类型
    object_type=[]  #客体类型
    predicate=[]    #两者关系
    with open("../data/all_50_schemas","r",encoding="utf8") as inp:
        for line in inp:
            line=json.loads(line.strip())
            subject_type.append(line["subject_type"])
            object_type.append(line["object_type"])
            predicate.append(line["predicate"])

    entity_type=list(set(subject_type+object_type))
    predicate_type=list(set(predicate))

    return entity_type,predicate_type



def convert_dict_to_df(input_path,output_path):
    """将json文件转换为Dataframe格式"""
    postag=[]
    predicate=[]
    with open(input_path,"r",encoding="utf8") as inp:
        for line in inp:  #every line
            line=json.loads(line.strip())  #['postag', 'text', 'spo_list']
            sentence=""
            obj_to_sub={} #[客体在句子中的位置，主体在句子中的位置，主客体两者之间的关系]
            for j in range(len(line["spo_list"])):  #every relation
                for i,token in enumerate(line["postag"]):  #every token
                    if token["word"]==line["spo_list"][j]["object"]:
                        if token["word"] not in sentence:
                            sentence+=token["word"]+"///"+token["pos"]+"-"+line["spo_list"][j]["object_type"]+"\t"
                        obj_to_sub["obj_index_%d"%j]=i
                    elif token["word"]==line["spo_list"][j]["subject"]:
                        if token["word"] not in sentence:
                            sentence+=token["word"]+"///"+token["pos"]+"-"+line["spo_list"][j]["subject_type"]+"\t"
                        obj_to_sub["sub_index_%d"%j]=i
                    else:
                        if token["word"] not in sentence:
                            sentence+=token["word"]+"///"+token["pos"]+"\t"
                    obj_to_sub["relation_%d"%j]=line["spo_list"][j]["predicate"]

            # print(sentence)
            # exit()
            postag.append(sentence.strip())
            predicate.append(obj_to_sub)

    dict={"postag":pd.Series(postag),
          "relations":pd.Series(predicate)}
    df=pd.DataFrame(dict)

    #保存
    df.to_csv(output_path,index=False)

    return df


def ner_process(input_path,output_path):
    """BIO标注"""
    df=pd.read_csv(input_path).values
    output=open(output_path,"w",encoding="utf8")

    for index,line in enumerate(df):
        tokens=str(line[0]).split("\t")
        # relation= json.loads(line[1])
        for token in tokens:
            word=token.split("///")[0] #词
            try:
                pos=token.split("///")[1]  #词性  部分句子没有分词和词性标注
            except IndexError:
                pass

            if "-" in pos:
                """对于要识别的实体"""
                for ids in range(len(word)):
                    if ids==0:
                        output.write(word[ids]+" B-"+pos.split("-")[1]+"\n")
                    else:
                        output.write(word[ids]+" I-"+pos.split("-")[1]+"\n")
            else:
                for char in word:
                    output.write(char+" O"+"\n")

        output.write("\n")

    output.close()















if __name__ == '__main__':
    df=convert_dict_to_df("../data/train_data.json","../data/train_data.csv")
    ner_process("../data/train_data.csv","../data/train_data_ner.txt")

    df = convert_dict_to_df("../data/dev_data.json", "../data/dev_data.csv")
    ner_process("../data/dev_data.csv", "../data/dev_data_ner.txt")

    # index=0
    # with open("../data/train_data_ner.txt","r",encoding="utf8") as inp:
    #     for line in inp:
    #         index+=1
    #         line=line.rstrip().split()
    #         if len(line)>1:
    #             if line[1]=="B-seijin":
    #                 print(index)
    #                 print(line)
    #         else:
    #             continue
