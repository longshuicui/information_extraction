import json
from tqdm import tqdm

#关系映射字典
relation2id={"N":0}
id2relation={0:"N"}
with open("../data/all_50_schemas","r",encoding="utf8") as inp:
    for line in inp:
        line=json.loads(line)
        relation2id[line["predicate"]]=len(relation2id)
for k,v in relation2id.items():
    id2relation[v]=k
with open("./all_50_schemas.json","w",encoding="utf8") as outp:
    json.dump([relation2id,id2relation],outp,indent=4,ensure_ascii=False)
exit()

char_counts={}  #统计字出现的次数
train_data=[]
with open("../data/train_data.json","r",encoding="utf8") as inp:
    for line in inp:
        line=json.loads(line)
        for char in line["text"]:
            char_counts[char]=char_counts.get(char,0)+1
        train_data.append({"text":line["text"],"spo_list":line["spo_list"]})

dev_data=[]
with open("../data/dev_data.json","r",encoding="utf8") as inp:
    for line in inp:
        line=json.loads(line)
        for char in line["text"]:
            char_counts[char]=char_counts.get(char,0)+1
        dev_data.append({"text":line["text"],"spo_list":line["spo_list"]})

test_data=[]
with open("../data/test1_data_postag.json","r",encoding="utf8") as inp:
    for line in inp:
        line=json.loads(line)
        for char in line["text"]:
            char_counts[char]=char_counts.get(char,0)+1
        test_data.append({"text":line["text"],"spo_list":[]})

char2id={}
id2char={}
for char,_ in char_counts.items():
    char2id[char]=len(char2id)+2
for k,v in char2id.items():
    id2char[v]=k

with open("./all_char_dict.json","w",encoding="utf8") as outp:
    json.dump([char2id,id2char],outp,indent=4,ensure_ascii=False)

with open("./train_data.json","w",encoding="utf8") as outp:
    json.dump(train_data,outp,indent=4,ensure_ascii=False)

with open("./dev_data.json","w",encoding="utf8") as outp:
    json.dump(dev_data,outp,indent=4,ensure_ascii=False)

with open("./test_data.json","w",encoding="utf8") as outp:
    json.dump(test_data,outp,indent=4,ensure_ascii=False)

