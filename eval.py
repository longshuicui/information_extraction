import utils
import copy
import numpy as np
from prettytable import PrettyTable


"""
Set of classes and methods for computing and printing the results using the strict, boundaries and relaxed evaluation methods.
For more info about how to use them see tf_utils.py
"""

class printClasses():
    def __init__(self):
        self.t = PrettyTable(['Class', 'TP', 'FP', 'FN', 'Pr', 'Re', 'F1'])

    def add(self, Class, TP, FP, FN, Pr, Re, F1):
        if Class!="O":
            self.t.add_row([Class, TP, FP, FN, Pr, Re, F1])

    def print(self):
        print(self.t)


def get_chunk_type(tok, idx_to_tag):
    # method implemented in https://github.com/guillaumegenthial/sequence_tagging/blob/master/model/data_utils.py
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    # method implemented in https://github.com/guillaumegenthial/sequence_tagging/blob/master/model/data_utils.py
    """Given a sequence of tags, group entities and their position
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """

    default = tags['O']
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i-1)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i-1)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq)-1)
        chunks.append(chunk)

    return chunks



def relationChunks(relations, ners, relationTuple="boundaries_type"):
    relationChunks = []
    for rel in relations:

        relation = rel[1]
        left_chunk = ""
        right_chunk = ""
        for ner in ners:
            if rel[0] >= ner[1] and rel[0] <= ner[2]:
                # print (ner)
                if relationTuple == "boundaries_type":
                    left_chunk = ner
                elif relationTuple == "boundaries":
                    left_chunk = (ner[1], ner[2])
                elif relationTuple == "type":
                    left_chunk = (ner[0])
            if rel[2] >= ner[1] and rel[2] <= ner[2]:
                # print (ner)
                if relationTuple == "boundaries_type":
                    right_chunk = ner
                elif relationTuple == "boundaries":
                    right_chunk = (ner[1], ner[2])
                elif relationTuple == "type":
                    right_chunk = (ner[0])
        if (left_chunk != "" and right_chunk != ""):
            relationChunks.append((left_chunk, relation, right_chunk))
    return relationChunks

def getTokenRelations(label_names, head_ids, token_ids):
        relations = []
        for labelLIdx in range(len(label_names)):
            # print (predLabel)
            labelL = label_names[labelLIdx]
            headL = head_ids[labelLIdx]
            tokenId = token_ids[labelLIdx]
            for labelIdx in range(len(labelL)):

                label = labelL[labelIdx]
                head = headL[labelIdx]
                # print (label)
                # print ((tokenId)+" "+ label+ " " + head)
                if label != "N":
                    # print (label)
                    relations.append((tokenId, label, head))
                    # print (tokenId,label,head)
        return relations


def keepOnlyChunkBoundaries(ners):
    nersNoBounds = []
    ners = list(ners)
    for ner in ners:
        # ner[0]=None
        # print (ner)
        nersNoBounds.append((None, ner[1], ner[2]))
    return nersNoBounds

class chunkEvaluator:
    def __init__(self,config,ner_chunk_eval="boundaries_type",rel_chunk_eval="boundaries"):
        ner_label=[ 'O','B-出版社', 'I-出版社', 'B-历史人物', 'I-历史人物', 'B-学科专业', 'I-学科专业', 'B-企业', 'I-企业', 'B-Date',
                    'I-Date', 'B-音乐专辑', 'I-音乐专辑', 'B-机构', 'I-机构', 'B-图书作品', 'I-图书作品', 'B-生物', 'I-生物',
                    'B-Text', 'I-Text', 'B-学校', 'I-学校', 'B-作品', 'I-作品', 'B-目', 'I-目', 'B-人物', 'I-人物',
                    'B-影视作品', 'I-影视作品', 'B-歌曲', 'I-歌曲', 'B-书籍', 'I-书籍', 'B-城市', 'I-城市', 'B-语言', 'I-语言',
                    'B-网络小说', 'I-网络小说', 'B-国家', 'I-国家', 'B-地点', 'I-地点', 'B-气候', 'I-气候', 'B-景点', 'I-景点',
                    'B-网站', 'I-网站', 'B-Number', 'I-Number', 'B-电视综艺', 'I-电视综艺', 'B-行政区', 'I-行政区']
        rel_label=['N', '朝代', '出版社', '目', '出生地', '妻子', '专业代码', '作曲', '身高', '作者', '字',
                   '出品公司', '面积', '首都', '作词', '号', '歌手', '毕业院校', '母亲', '连载网站', '主持人',
                   '创始人', '改编自', '海拔', '简称', '主角', '邮政编码', '气候', '民族', '占地面积', '总部地点',
                   '人口数量', '制片人', '丈夫', '出生日期', '祖籍', '嘉宾', '官方语言', '父亲', '国籍', '所在城市',
                   '上映时间', '注册资本', '导演', '修业年限', '编剧', '主演', '董事长', '成立日期', '所属专辑']

        config.dataset_set_ec_tags=set()
        self.nerSegmentationTags=ner_label

        self.NERset = config.dataset_set_ec_tags
        self.RELset = rel_label

        self.root_node=False


        self.ner_chunk_eval=ner_chunk_eval
        self.rel_chunk_eval=rel_chunk_eval


        self.totals = 0
        self.oks = 0

        self.tpsNER = 0
        self.fpsNER = 0
        self.fnsNER = 0

        self.tpsREL = 0
        self.fpsREL = 0
        self.fnsREL = 0

        self.tpsClassesNER = dict.fromkeys(self.NERset, 0)
        self.fpsClassesNER = dict.fromkeys(self.NERset, 0)
        self.fnsClassesNER = dict.fromkeys(self.NERset, 0)
        self.precisionNER = dict.fromkeys(self.NERset, 0)
        self.recallNER = dict.fromkeys(self.NERset, 0)
        self.f1NER = dict.fromkeys(self.NERset, 0)

        self.tpsClassesREL = dict.fromkeys(self.RELset, 0)
        self.fpsClassesREL = dict.fromkeys(self.RELset, 0)
        self.fnsClassesREL = dict.fromkeys(self.RELset, 0)

        self.precisionREL = dict.fromkeys(self.RELset, 0)
        self.recallREL = dict.fromkeys(self.RELset, 0)
        self.f1REL = dict.fromkeys(self.RELset, 0)

        self.correct_predsNER, self.total_correctNER, self.total_predsNER = 0., 0., 0.
        self.correct_predsREL, self.total_correctREL, self.total_predsREL = 0., 0., 0.

    def add(self, pred_batchesNER, true_batchesNER, pred_batchesREL, true_batchesREL):



        for batch_idx in range(len(pred_batchesNER)):
            predNER = pred_batchesNER[batch_idx]
            trueNER = true_batchesNER[batch_idx]

            predRel = pred_batchesREL[batch_idx]
            trueRel = true_batchesREL[batch_idx]

            ptoken_ids, _, plabel_ids, phead_ids, plabel_names = transformToInitialInput(
                predRel, self.RELset)

            _, _, tlabel_ids, thead_ids, tlabel_names = transformToInitialInput(
                trueRel, self.RELset)

            trueRel = getTokenRelations(tlabel_names, thead_ids, ptoken_ids)

            predRel = getTokenRelations(plabel_names, phead_ids, ptoken_ids)

            tagsNER = getSegmentationDict(self.nerSegmentationTags)#self.

            if self.ner_chunk_eval == "boundaries_type":

                lab_chunks = set(get_chunks(trueNER, tagsNER))
                lab_pred_chunks = set(get_chunks(predNER, tagsNER))

            elif self.ner_chunk_eval == "boundaries":

                lab_chunks = set(keepOnlyChunkBoundaries(set(get_chunks(trueNER, tagsNER))))
                lab_pred_chunks = set(keepOnlyChunkBoundaries(set(get_chunks(predNER, tagsNER))))




            lab_chunks_list = list(lab_chunks)
            lab_pred_chunks_list = list(lab_pred_chunks)


            if self.ner_chunk_eval == "boundaries_type":
                for lab_idx in range(len(lab_pred_chunks_list)):

                    if lab_pred_chunks_list[lab_idx] in lab_chunks_list:
                        # print (lab_pred_chunks_list[lab_idx][0])
                        self.tpsClassesNER[lab_pred_chunks_list[lab_idx][0]] += 1
                    else:
                        self.fpsClassesNER[lab_pred_chunks_list[lab_idx][0]] += 1
                        # fnsEntitiesNER+=1

                for lab_idx in range(len(lab_chunks_list)):

                    if lab_chunks_list[lab_idx] not in lab_pred_chunks_list:
                        self.fnsClassesNER[lab_chunks_list[lab_idx][0]] += 1

            elif self.ner_chunk_eval == "boundaries":
                for lab_idx in range(len(lab_pred_chunks_list)):

                    if lab_pred_chunks_list[lab_idx] in lab_chunks_list:
                        # print (lab_pred_chunks_list[lab_idx][0])
                        self.tpsNER += 1
                    else:
                        self.fpsNER += 1
                        # fnsEntitiesNER+=1

                for lab_idx in range(len(lab_chunks_list)):

                    if lab_chunks_list[lab_idx] not in lab_pred_chunks_list:
                        self.fnsNER += 1

            if self.root_node==True:
                lab_chunks_list_with_ROOT = copy.deepcopy(lab_chunks_list)
                lab_chunks_list_with_ROOT.append((None,0, 0))

                lab_pred_chunks_list_with_ROOT = copy.deepcopy(lab_pred_chunks_list)
                lab_pred_chunks_list_with_ROOT.append((None,0, 0))

                relTrue = set(relationChunks(trueRel, lab_chunks_list_with_ROOT, relationTuple=self.rel_chunk_eval))

                relPred = set(relationChunks(predRel, lab_pred_chunks_list_with_ROOT, relationTuple=self.rel_chunk_eval))

            else:
                relTrue = set(relationChunks(trueRel, lab_chunks_list,relationTuple=self.rel_chunk_eval))

                relPred = set(relationChunks(predRel, lab_pred_chunks_list,relationTuple=self.rel_chunk_eval))

            relTrueList = list(relTrue)  # trueRel#

            # if (len(trueRel)!=len(relTrueList)):
            #    print ("warning")

            relPredList = list(relPred)  # predRel#

            for lab_idx in range(len(relPredList)):

                if relPredList[lab_idx] in relTrueList:
                    # print (lab_pred_chunks_list[lab_idx][0])
                    self.tpsClassesREL[relPredList[lab_idx][1]] += 1
                    # print (relPredList[lab_idx])
                else:
                    self.fpsClassesREL[relPredList[lab_idx][1]] += 1
                    # fnsEntitiesNER+=1

            for lab_idx in range(len(relTrueList)):

                if relTrueList[lab_idx] not in relPredList:
                    self.fnsClassesREL[relTrueList[lab_idx][1]] += 1

            self.correct_predsNER += len(lab_chunks & lab_pred_chunks)
            self.total_predsNER += len(lab_pred_chunks)
            self.total_correctNER += len(lab_chunks)

            self.correct_predsREL += len(relTrue & relPred)
            self.total_predsREL += len(relPred)
            self.total_correctREL += len(relTrue)



    def getResultsNER(self):
        p = self.correct_predsNER / self.total_predsNER if self.correct_predsNER > 0 else 0
        r = self.correct_predsNER / self.total_correctNER if self.correct_predsNER > 0 else 0
        f1 = 2 * p * r / (p + r) if self.correct_predsNER > 0 else 0

        print(self.correct_predsNER)
        print(self.total_predsNER)
        print(self.total_correctNER)

        print(f1)
        return f1

    def getResultsREL(self):
        p = self.correct_predsREL / self.total_predsREL if self.correct_predsREL > 0 else 0
        r = self.correct_predsREL / self.total_correctREL if self.correct_predsREL > 0 else 0
        f1 = 2 * p * r / (p + r) if self.correct_predsREL > 0 else 0

        print(self.correct_predsREL)
        print(self.total_predsREL)
        print(self.total_correctREL)

        print(f1)
        return f1

    def getPrecision(self, tps, fps):
        if tps == 0:
            return 0
        else:
            return tps / (tps + fps)

    def getRecall(self, tps, fns):
        if tps == 0:
            return 0
        else:
            return tps / (tps + fns)

    def getF1(self, tps, fps, fns):
        if tps == 0:
            return 0
        else:
            return 2 * self.getPrecision(tps, fps) * self.getRecall(tps, fns) / (
            self.getPrecision(tps, fps) + self.getRecall(tps, fns))


    def getChunkedOverallAvgF1(self):


        return (self.getChunkedNERF1()+self.getChunkedRELF1())/2

    def getChunkedOverallF1(self):
        tpsNER=0
        fnsNER=0
        fpsNER=0
        tpsREL=0
        fnsREL=0
        fpsREL=0
        if self.ner_chunk_eval == "boundaries_type":
            for label in self.NERset:
                # if label != "O" :
                tpsNER += self.tpsClassesNER[label]

                fnsNER += self.fnsClassesNER[label]
                fpsNER += self.fpsClassesNER[label]
        elif self.ner_chunk_eval == "boundaries":
            tpsNER=self.tpsNER
            fnsNER = self.fnsNER
            fpsNER = self.fpsNER


        for label in self.RELset:

            if label != "N":
                tpsREL += self.tpsClassesREL[label]

                fnsREL += self.fnsClassesREL[label]
                fpsREL += self.fpsClassesREL[label]



        return self.getF1(tpsNER+tpsREL, fpsNER+fpsREL, fnsNER+fnsREL)


    def getOverallF1(self):
        tpsNER=0
        fnsNER=0
        fpsNER=0
        tpsREL=0
        fnsREL=0
        fpsREL=0

        for label in self.NERset:
            # if label != "O" :
            tpsNER += self.tpsClassesNER[label]

            fnsNER += self.fnsClassesNER[label]
            fpsNER += self.fpsClassesNER[label]

        for label in self.RELset:

            if label != "N":
                tpsREL += self.tpsClassesREL[label]

                fnsREL += self.fnsClassesREL[label]
                fpsREL += self.fpsClassesREL[label]



        return self.getF1(tpsNER+tpsREL, fpsNER+fpsREL, fnsNER+fnsREL)

    def getChunkedRELF1(self):

        tpsREL=0
        fnsREL=0
        fpsREL=0



        for label in self.RELset:

            if label != "N":
                tpsREL += self.tpsClassesREL[label]

                fnsREL += self.fnsClassesREL[label]
                fpsREL += self.fpsClassesREL[label]



        return self.getF1(tpsREL, fpsREL, fnsREL)

    def getChunkedNERF1(self):
        tpsNER = 0
        fnsNER = 0
        fpsNER = 0
        if self.ner_chunk_eval == "boundaries_type":


            for label in self.NERset:
                # if label != "O" :
                tpsNER += self.tpsClassesNER[label]

                fnsNER += self.fnsClassesNER[label]
                fpsNER += self.fpsClassesNER[label]


        elif self.ner_chunk_eval== "boundaries":
            tpsNER =self.tpsNER
            fnsNER = self.fnsNER
            fpsNER = self.fpsNER

        return self.getF1(tpsNER, fpsNER, fnsNER)
    def getAccuracy(self):
        return self.oks / self.totals

    def printInfo(self):

        printer = printClasses()

        if self.ner_chunk_eval== "boundaries_type":
            for label in self.NERset:
                # if label != "O" :
                self.tpsNER += self.tpsClassesNER[label]

                self.fnsNER += self.fnsClassesNER[label]
                self.fpsNER += self.fpsClassesNER[label]

                printer.add(label, self.tpsClassesNER[label], self.fpsClassesNER[label], self.fnsClassesNER[label],
                            self.getPrecision(self.tpsClassesNER[label], self.fpsClassesNER[label]),
                            self.getRecall(self.tpsClassesNER[label], self.fnsClassesNER[label]),
                            self.getF1(self.tpsClassesNER[label], self.fpsClassesNER[label], self.fnsClassesNER[label]))



                # print('%s TP: %d  FP: %d  FN: %d TN: %d precision: %f recall: %f F1: %f' % (label,self.tpsClasses[label],self.fpsClasses[label],self.fnsClasses[label],self.tnsClasses[label], self.precision[label], self.recall[label], self.f1[label]))
            printer.add("-", "-", "-", "-",
                        "-", "-",
                        "-")
            printer.add("Micro NER chunk", self.tpsNER, self.fpsNER, self.fnsNER,
                    self.getPrecision(self.tpsNER, self.fpsNER), self.getRecall(self.tpsNER, self.fnsNER),
                    self.getF1(self.tpsNER, self.fpsNER, self.fnsNER))

        elif self.ner_chunk_eval== "boundaries":
            printer.add("Micro NER chunk boundaries", self.tpsNER, self.fpsNER, self.fnsNER,
                        self.getPrecision(self.tpsNER, self.fpsNER), self.getRecall(self.tpsNER, self.fnsNER),
                        self.getF1(self.tpsNER, self.fpsNER, self.fnsNER))
        printer.print()

        printer = printClasses()
        for label in self.RELset:

            if label != "N":
                self.tpsREL += self.tpsClassesREL[label]

                self.fnsREL += self.fnsClassesREL[label]
                self.fpsREL += self.fpsClassesREL[label]

                printer.add(label, self.tpsClassesREL[label], self.fpsClassesREL[label], self.fnsClassesREL[label],
                            self.getPrecision(self.tpsClassesREL[label], self.fpsClassesREL[label]),
                            self.getRecall(self.tpsClassesREL[label], self.fnsClassesREL[label]),
                            self.getF1(self.tpsClassesREL[label], self.fpsClassesREL[label], self.fnsClassesREL[label]))



                # print('%s TP: %d  FP: %d  FN: %d TN: %d precision: %f recall: %f F1: %f' % (label,self.tpsClasses[label],self.fpsClasses[label],self.fnsClasses[label],self.tnsClasses[label], self.precision[label], self.recall[label], self.f1[label]))
        printer.add("-", "-", "-", "-",
                    "-", "-",
                    "-")
        printer.add("Micro REL chunk", self.tpsREL, self.fpsREL, self.fnsREL,
                    self.getPrecision(self.tpsREL, self.fpsREL), self.getRecall(self.tpsREL, self.fnsREL),
                    self.getF1(self.tpsREL, self.fpsREL, self.fnsREL))

        printer.print()


def getMaxOccurence(lst):
    from collections import Counter
    most_common, num_most_common = Counter(lst).most_common(1)[0]  # 4, 6 times
    return most_common


def classesToChunks(tokenClasses, chunks):
    labeled_chunks = []
    for chunk in chunks:

        class_list = (tokenClasses[chunk[1]:chunk[2] + 1])

        if chunk[0] in class_list:
            labeled_chunks.append((chunk[0], chunk[1], chunk[2]))
        else:
            labeled_chunks.append((getMaxOccurence(class_list), chunk[1], chunk[2]))
            # print (class_list)
    return labeled_chunks


def listOfTagsToids(lstTags,tags):
    lstids = []
    for ner in lstTags:
        lstids.append(tags.index(ner))

    return lstids

def listOfIdsToTags(lst_ids,tags):
    lstTags= []
    for nerId in lst_ids:
        lstTags.append(tags[nerId])
    return lstTags

class relaxedChunkEvaluator:
    def __init__(self,dataset_params,rel_chunk_eval="boundaries"):
        self.nerSegmentationTags=dataset_params.dataset_set_bio_tags

        self.NERset = dataset_params.dataset_set_ec_tags#utils.getNerSetACE04()
        self.RELset = dataset_params.dataset_set_relations#reutils.getRelSetACE04()
        #self.nerDict=dataset_params
        # print (self.NERset)
        self.rel_chunk_eval=rel_chunk_eval
        self.totals = 0
        self.oks = 0

        self.tpsNER = 0
        self.fpsNER = 0
        self.fnsNER = 0

        self.tpsREL = 0
        self.fpsREL = 0
        self.fnsREL = 0

        self.tpsNERMacro = 0
        self.fpsNERMacro = 0
        self.fnsNERMacro = 0

        self.tpsNERMacro_no_other = 0
        self.fpsNERMacro_no_other = 0
        self.fnsNERMacro_no_other = 0

        self.tpsRELMacro = 0
        self.fpsRELMacro = 0
        self.fnsRELMacro = 0


        self.NERF1Macro=0
        self.NERF1Macro_no_other = 0
        self.RELF1Macro = 0
        self.OverallF1Macro = 0
        self.OverallF1Macro_no_other  = 0


        self.tpsClassesNER = dict.fromkeys(self.NERset, 0)
        self.fpsClassesNER = dict.fromkeys(self.NERset, 0)
        self.fnsClassesNER = dict.fromkeys(self.NERset, 0)
        self.precisionNER = dict.fromkeys(self.NERset, 0)
        self.recallNER = dict.fromkeys(self.NERset, 0)
        self.f1NER = dict.fromkeys(self.NERset, 0)

        self.tpsClassesREL = dict.fromkeys(self.RELset, 0)
        self.fpsClassesREL = dict.fromkeys(self.RELset, 0)
        self.fnsClassesREL = dict.fromkeys(self.RELset, 0)

        self.precisionREL = dict.fromkeys(self.RELset, 0)
        self.recallREL = dict.fromkeys(self.RELset, 0)
        self.f1REL = dict.fromkeys(self.RELset, 0)

        self.correct_predsNER, self.total_correctNER, self.total_predsNER = 0., 0., 0.
        self.correct_predsREL, self.total_correctREL, self.total_predsREL = 0., 0., 0.

    def add(self, pred_batchesNER, true_batchesNER, pred_batchesREL, true_batchesREL,true_batchesBIONER):



        for batch_idx in range(len(pred_batchesNER)):
            predNER = pred_batchesNER[batch_idx]
            trueNER = true_batchesNER[batch_idx]

            predRel = pred_batchesREL[batch_idx]
            trueRel = true_batchesREL[batch_idx]

            trueBIONER=true_batchesBIONER[batch_idx]


            ptoken_ids, _, plabel_ids, phead_ids, plabel_names = utils.transformToInitialInput(
                predRel, self.RELset)

            _, _, tlabel_ids, thead_ids, tlabel_names = utils.transformToInitialInput(
                trueRel, self.RELset)

            trueRel = getTokenRelations(tlabel_names, thead_ids, ptoken_ids)

            predRel = getTokenRelations(plabel_names, phead_ids, ptoken_ids)


            #print (self.NERset)
            tagsNER = utils.getSegmentationDict(self.nerSegmentationTags)#self.



            lab_chunks_ = set(get_chunks(listOfTagsToids(trueBIONER,self.nerSegmentationTags), tagsNER))
            #lab_pred_chunks = set(get_chunks(predNER, tagsNER))

            lab_chunks_list_ = list(lab_chunks_)


            trueNER_tags=listOfIdsToTags(trueNER,self.NERset)
            predNER_tags=listOfIdsToTags(predNER, self.NERset)

            lab_chunks = set(classesToChunks(trueNER_tags, lab_chunks_list_))
            lab_pred_chunks=set(classesToChunks(predNER_tags, lab_chunks_list_))

            lab_chunks_list = list(lab_chunks)
            lab_pred_chunks_list = list(lab_pred_chunks)


            for lab_idx in range(len(lab_pred_chunks_list)):

                if lab_pred_chunks_list[lab_idx] in lab_chunks_list:
                    # print (lab_pred_chunks_list[lab_idx][0])
                    self.tpsClassesNER[lab_pred_chunks_list[lab_idx][0]] += 1
                else:
                    self.fpsClassesNER[lab_pred_chunks_list[lab_idx][0]] += 1
                    # fnsEntitiesNER+=1

            for lab_idx in range(len(lab_chunks_list)):

                if lab_chunks_list[lab_idx] not in lab_pred_chunks_list:
                    self.fnsClassesNER[lab_chunks_list[lab_idx][0]] += 1

            relTrue = set(relationChunks(trueRel, lab_chunks_list,relationTuple=self.rel_chunk_eval))

            relPred = set(relationChunks(predRel, lab_pred_chunks_list,relationTuple=self.rel_chunk_eval))

            relTrueList = list(relTrue)  # trueRel#

            # if (len(trueRel)!=len(relTrueList)):
            #    print ("warning")

            relPredList = list(relPred)  # predRel#

            #print("GOLD REL chunks:" + str(relTrueList))

            #print("PRED REL chunks:" + str(relPredList))

            for lab_idx in range(len(relPredList)):

                if relPredList[lab_idx] in relTrueList:
                    # print (lab_pred_chunks_list[lab_idx][0])
                    self.tpsClassesREL[relPredList[lab_idx][1]] += 1
                    # print (relPredList[lab_idx])
                else:
                    self.fpsClassesREL[relPredList[lab_idx][1]] += 1
                    # fnsEntitiesNER+=1

            for lab_idx in range(len(relTrueList)):

                if relTrueList[lab_idx] not in relPredList:
                    self.fnsClassesREL[relTrueList[lab_idx][1]] += 1

            self.correct_predsNER += len(lab_chunks & lab_pred_chunks)
            self.total_predsNER += len(lab_pred_chunks)
            self.total_correctNER += len(lab_chunks)

            self.correct_predsREL += len(relTrue & relPred)
            self.total_predsREL += len(relPred)
            self.total_correctREL += len(relTrue)



    def getResultsNER(self):
        p = self.correct_predsNER / self.total_predsNER if self.correct_predsNER > 0 else 0
        r = self.correct_predsNER / self.total_correctNER if self.correct_predsNER > 0 else 0
        f1 = 2 * p * r / (p + r) if self.correct_predsNER > 0 else 0

        print(self.correct_predsNER)
        print(self.total_predsNER)
        print(self.total_correctNER)

        print(f1)
        return f1

    def getResultsREL(self):
        p = self.correct_predsREL / self.total_predsREL if self.correct_predsREL > 0 else 0
        r = self.correct_predsREL / self.total_correctREL if self.correct_predsREL > 0 else 0
        f1 = 2 * p * r / (p + r) if self.correct_predsREL > 0 else 0

        print(self.correct_predsREL)
        print(self.total_predsREL)
        print(self.total_correctREL)

        print(f1)
        return f1

    def getPrecision(self, tps, fps):
        if tps == 0:
            return 0
        else:
            return tps / (tps + fps)

    def getRecall(self, tps, fns):
        if tps == 0:
            return 0
        else:
            return tps / (tps + fns)

    def getF1(self, tps, fps, fns):
        if tps == 0:
            return 0
        else:
            return 2 * self.getPrecision(tps, fps) * self.getRecall(tps, fns) / (
            self.getPrecision(tps, fps) + self.getRecall(tps, fns))

    def getChunkedOverallF1(self):
        tpsNER=0
        fnsNER=0
        fpsNER=0
        tpsREL=0
        fnsREL=0
        fpsREL=0

        for label in self.NERset:
            # if label != "O" :
            tpsNER += self.tpsClassesNER[label]

            fnsNER += self.fnsClassesNER[label]
            fpsNER += self.fpsClassesNER[label]

        for label in self.RELset:

            if label != "N":
                tpsREL += self.tpsClassesREL[label]

                fnsREL += self.fnsClassesREL[label]
                fpsREL += self.fpsClassesREL[label]



        return self.getF1(tpsNER+tpsREL, fpsNER+fpsREL, fnsNER+fnsREL)


    def getOverallF1(self):
        tpsNER=0
        fnsNER=0
        fpsNER=0
        tpsREL=0
        fnsREL=0
        fpsREL=0

        for label in self.NERset:
            # if label != "O" :
            tpsNER += self.tpsClassesNER[label]

            fnsNER += self.fnsClassesNER[label]
            fpsNER += self.fpsClassesNER[label]

        for label in self.RELset:

            if label != "N":
                tpsREL += self.tpsClassesREL[label]

                fnsREL += self.fnsClassesREL[label]
                fpsREL += self.fpsClassesREL[label]



        return self.getF1(tpsNER+tpsREL, fpsNER+fpsREL, fnsNER+fnsREL)

    def getChunkedRELF1(self):

        tpsREL=0
        fnsREL=0
        fpsREL=0



        for label in self.RELset:

            if label != "N":
                tpsREL += self.tpsClassesREL[label]

                fnsREL += self.fnsClassesREL[label]
                fpsREL += self.fpsClassesREL[label]



        return self.getF1(tpsREL, fpsREL, fnsREL)

    def getChunkedNERF1(self):
        tpsNER=0
        fnsNER=0
        fpsNER=0


        for label in self.NERset:
            # if label != "O" :
            tpsNER += self.tpsClassesNER[label]

            fnsNER += self.fnsClassesNER[label]
            fpsNER += self.fpsClassesNER[label]

        return self.getF1(tpsNER, fpsNER, fnsNER)

    def getAccuracy(self):
        return self.oks / self.totals

    def getMacroF1scores(self):


        return self.NERF1Macro,self.RELF1Macro,self.OverallF1Macro

    def getMacroF1scoresNoOtherClass(self):

        return self.NERF1Macro_no_other, self.RELF1Macro, self.OverallF1Macro_no_other


    def computeInfoMacro(self,printScores=True):

        printer = printClasses()


        averageNERF1_no_Other=0
        averageNERF1 = 0

        averageNERrecall_no_Other = 0
        averageNERrecall = 0

        averageNERprecision_no_Other = 0
        averageNERprecision = 0

        for label in self.NERset:
            if label != "O":
                self.tpsNERMacro += self.tpsClassesNER[label]

                self.fnsNERMacro += self.fnsClassesNER[label]
                self.fpsNERMacro += self.fpsClassesNER[label]

            f1_class=self.getF1(self.tpsClassesNER[label], self.fpsClassesNER[label], self.fnsClassesNER[label])
            precision_class=self.getPrecision(self.tpsClassesNER[label], self.fpsClassesNER[label])
            recall_class=self.getRecall(self.tpsClassesNER[label], self.fnsClassesNER[label])
            if label!= "O" :
                averageNERF1+=f1_class
                averageNERrecall += recall_class
                averageNERprecision += precision_class

            if label!= "O" and label!= "Other":
                averageNERF1_no_Other+=f1_class
                averageNERrecall_no_Other += recall_class
                averageNERprecision_no_Other += precision_class


            if label != "O" and label != "Other":
                self.tpsNERMacro_no_other += self.tpsClassesNER[label]

                self.fnsNERMacro_no_other += self.fnsClassesNER[label]
                self.fpsNERMacro_no_other += self.fpsClassesNER[label]


            printer.add(label, self.tpsClassesNER[label], self.fpsClassesNER[label], self.fnsClassesNER[label],
                        precision_class,
                        recall_class,
                        f1_class)



            # print('%s TP: %d  FP: %d  FN: %d TN: %d precision: %f recall: %f F1: %f' % (label,self.tpsClasses[label],self.fpsClasses[label],self.fnsClasses[label],self.tnsClasses[label], self.precision[label], self.recall[label], self.f1[label]))
        printer.add("-", "-", "-", "-",
                    "-", "-",
                    "-")

        averageNERrecall = averageNERrecall / (len(self.NERset) - 1)
        averageNERprecision = averageNERprecision / (len(self.NERset) - 1)
        averageNERF1 = averageNERF1 / (len(self.NERset) - 1)


        if "other" in [x.lower() for x in self.NERset]:

            averageNERprecision_no_Other=averageNERprecision_no_Other / (len(self.NERset) -2)
            averageNERrecall_no_Other=averageNERrecall_no_Other / (len(self.NERset) -2)
            averageNERF1_no_Other=averageNERF1_no_Other / (len(self.NERset) -2)

            printer.add("Macro NER chunk RELAXED ^Other", self.tpsNERMacro_no_other, self.fpsNERMacro_no_other, self.fnsNERMacro_no_other,
                        averageNERprecision_no_Other, averageNERrecall_no_Other,
                        averageNERF1_no_Other)
        else:
            averageNERprecision_no_Other = averageNERprecision
            averageNERrecall_no_Other = averageNERrecall
            averageNERF1_no_Other = averageNERF1


        printer.add("Macro NER chunk RELAXED", self.tpsNERMacro, self.fpsNERMacro, self.fnsNERMacro,
                    averageNERprecision, averageNERrecall,
                    averageNERF1)
        if printScores ==True:

            printer.print()

        printer = printClasses()

        averageRELF1 = 0

        averageRELrecall = 0

        averageRELprecision = 0

        for label in self.RELset:

            if label != "N":
                self.tpsRELMacro += self.tpsClassesREL[label]

                self.fnsRELMacro += self.fnsClassesREL[label]
                self.fpsRELMacro += self.fpsClassesREL[label]

                f1_class = self.getF1(self.tpsClassesREL[label], self.fpsClassesREL[label], self.fnsClassesREL[label])
                precision_class = self.getPrecision(self.tpsClassesREL[label], self.fpsClassesREL[label])
                recall_class = self.getRecall(self.tpsClassesREL[label], self.fnsClassesREL[label])

                averageRELF1+=f1_class
                averageRELrecall += recall_class
                averageRELprecision += precision_class

                printer.add(label, self.tpsClassesREL[label], self.fpsClassesREL[label], self.fnsClassesREL[label],
                            precision_class,
                            recall_class,
                            f1_class)



                # print('%s TP: %d  FP: %d  FN: %d TN: %d precision: %f recall: %f F1: %f' % (label,self.tpsClasses[label],self.fpsClasses[label],self.fnsClasses[label],self.tnsClasses[label], self.precision[label], self.recall[label], self.f1[label]))
        printer.add("-", "-", "-", "-",
                    "-", "-",
                    "-")


        averageRELrecall=averageRELrecall/(len(self.RELset) - 1)
        averageRELprecision=averageRELprecision/(len(self.RELset) - 1)
        averageRELF1 =averageRELF1 /(len(self.RELset) - 1)




        printer.add("Macro REL chunk RELAXED", self.tpsRELMacro, self.fpsRELMacro, self.fnsRELMacro,
                    averageRELprecision, averageRELrecall,
                    averageRELF1)

        if printScores == True:
            printer.print()

        over_avg_f1 = (averageNERF1 + averageRELF1) / 2
        over_avg_f1_no_other = (averageNERF1_no_Other + averageRELF1) / 2

        t = PrettyTable(['Type','NER_F1', 'REL_F1', 'AVG_F1'])

        t.add_row(['Overall', averageNERF1, averageRELF1, over_avg_f1])
        if "other" in [x.lower() for x in self.NERset]:
            t.add_row(['Overall ^Other', averageNERF1_no_Other, averageRELF1, over_avg_f1_no_other])

        if printScores == True:
            print (t)

        self.NERF1Macro = averageNERF1
        self.NERF1Macro_no_other = averageNERF1_no_Other
        self.RELF1Macro = averageRELF1
        self.OverallF1Macro = over_avg_f1
        self.OverallF1Macro_no_other = over_avg_f1_no_other


    def printInfoMicro(self):

            printer = printClasses()

            for label in self.NERset:
                # if label != "O" :
                self.tpsNER += self.tpsClassesNER[label]

                self.fnsNER += self.fnsClassesNER[label]
                self.fpsNER += self.fpsClassesNER[label]

                printer.add(label, self.tpsClassesNER[label], self.fpsClassesNER[label], self.fnsClassesNER[label],
                            self.getPrecision(self.tpsClassesNER[label], self.fpsClassesNER[label]),
                            self.getRecall(self.tpsClassesNER[label], self.fnsClassesNER[label]),
                            self.getF1(self.tpsClassesNER[label], self.fpsClassesNER[label], self.fnsClassesNER[label]))



                # print('%s TP: %d  FP: %d  FN: %d TN: %d precision: %f recall: %f F1: %f' % (label,self.tpsClasses[label],self.fpsClasses[label],self.fnsClasses[label],self.tnsClasses[label], self.precision[label], self.recall[label], self.f1[label]))
            printer.add("-", "-", "-", "-",
                        "-", "-",
                        "-")
            printer.add("Micro NER chunk RELAXED", self.tpsNER, self.fpsNER, self.fnsNER,
                        self.getPrecision(self.tpsNER, self.fpsNER), self.getRecall(self.tpsNER, self.fnsNER),
                        self.getF1(self.tpsNER, self.fpsNER, self.fnsNER))

            printer.print()

            printer = printClasses()
            for label in self.RELset:

                if label != "N":
                    self.tpsREL += self.tpsClassesREL[label]

                    self.fnsREL += self.fnsClassesREL[label]
                    self.fpsREL += self.fpsClassesREL[label]

                    printer.add(label, self.tpsClassesREL[label], self.fpsClassesREL[label], self.fnsClassesREL[label],
                                self.getPrecision(self.tpsClassesREL[label], self.fpsClassesREL[label]),
                                self.getRecall(self.tpsClassesREL[label], self.fnsClassesREL[label]),
                                self.getF1(self.tpsClassesREL[label], self.fpsClassesREL[label],
                                           self.fnsClassesREL[label]))



                    # print('%s TP: %d  FP: %d  FN: %d TN: %d precision: %f recall: %f F1: %f' % (label,self.tpsClasses[label],self.fpsClasses[label],self.fnsClasses[label],self.tnsClasses[label], self.precision[label], self.recall[label], self.f1[label]))
            printer.add("-", "-", "-", "-",
                        "-", "-",
                        "-")
            printer.add("Micro REL chunk RELAXED", self.tpsREL, self.fpsREL, self.fnsREL,
                        self.getPrecision(self.tpsREL, self.fpsREL), self.getRecall(self.tpsREL, self.fnsREL),
                        self.getF1(self.tpsREL, self.fpsREL, self.fnsREL))

            printer.print()

def transformToInitialInput(matrix, tags):
    active_relations = np.nonzero(matrix)
    active_relations_iidx = active_relations[0]
    active_relations_jidx = active_relations[1]

    tokens_ids = []
    heads_ids = []
    labels_ids = []
    head_labels_ids = []
    labels_name = []

    for m_idx in range(len(matrix)):
        tokens_ids.append(m_idx)
        heads_ids.append([])
        labels_ids.append([])
        head_labels_ids.append([])
        labels_name.append([])

    for i_idx in range(len(active_relations_iidx)):
        head_id = int(active_relations_jidx[i_idx] / len(tags))
        label_id = active_relations_jidx[i_idx] % len(tags)
        token_id = active_relations_iidx[i_idx]
        head_label_id = active_relations_jidx[i_idx]

        # idx=tokens_ids.index(token_id)
        heads_ids[token_id].append(head_id)
        labels_ids[token_id].append(label_id)
        head_labels_ids[token_id].append(head_label_id)
        labels_name[token_id].append(tags[label_id])

        # print (str(token_id) + " " +str(head_label_id)+ " " +str(head)+ " " +str(label))
    return tokens_ids, head_labels_ids, labels_ids, heads_ids, labels_name


def getSegmentationDict(lst):
    return {k: v for v, k in enumerate(lst)}