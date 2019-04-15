information_extraction
实体识别和信息抽取 数据来源于--2019语言与智能技术竞赛信息抽取-- 链接地址：http://lic2019.ccf.org.cn/kg

convert_data.py  转换训练和验证数据集  
convert_test.py  转换测试集数据
eval,py   比赛中用的评估脚本
model.py   tensorflow下的模型
process.py  模型训练验证测试中用函数
utils.py    数据处理用的函数
vec.txt     预训练的中文字向量

all_50_schemas_me.json    relation2id id2relation
all_chars_me.json         char2id id2char
all_ner_label_me.json      label2id id2label
embedding.pkl            字向量
  

main.py   运行函数


代码中评测过程没有用到比赛的评估脚本

代码有些bug，希望有缘人指正。