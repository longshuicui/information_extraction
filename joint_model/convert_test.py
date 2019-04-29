import json
with open("test1_data_postag.json","r",encoding="utf8") as inp,open("test_data_me.json","w",encoding="utf8") as outp:
    test_data=[]
    for line in inp:
        line=json.loads(line)
        text=line["text"]
        test_data.append({"text":text})

    json.dump(test_data,outp,ensure_ascii=False,indent=4)