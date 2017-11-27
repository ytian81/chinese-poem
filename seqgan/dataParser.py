import sys
import os
import json
import re

# data process code adapted from https://github.com/justdark/pytorch-poetry-gen
def parseRawData(author = None, constrains = None, max_len = None):
    rst = []

    def sentenceParse(para):
        result, number = re.subn("（.*）", "", para)
        result, number = re.subn("（.*）", "", para)
        result, number = re.subn("{.*}", "", result)
        result, number = re.subn("《.*》", "", result)
        result, number = re.subn("《.*》", "", result)
        result, number = re.subn("[\]\[]", "", result)
        r = ""
        for s in result:
            if s not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']:
                r += s;
        r, number = re.subn("。。", "。", r)
        return r

    def handleJson(file):
        # print file
        rst = []
        data = json.loads(open(file).read())
        for poetry in data:
            pdata = ""
            if (author!=None and poetry.get("author")!=author):
                continue
            p = poetry.get("paragraphs")

            flag = False
            if constrains is not None:
                for s in p:
                    sp = re.split("[，！。]", s)
                    for tr in sp:
                        flag_num = 0
                        for constrain in constrains:
                            if constrain != None and len(tr) != constrain and len(tr)!=0:
                                flag_num += 1
                        if flag_num == len(constrains):
                            flag = True
                            break
                    if flag:
                        break
            if flag:
                continue
            for sentence in poetry.get("paragraphs"):
                sp = re.split("[！。]", sentence)
                for tr in sp:
                    if max_len is not None:
                        if len(tr) > max_len:
                            continue
                        if len(pdata) + len(tr) > max_len and len(pdata) > 11 and len(pdata) <= max_len:
                            rst.append(pdata)
                            pdata = ""
                    pdata += tr
                    pdata = sentenceParse(pdata)

            if len(pdata) > 11 and len(pdata) <= max_len:
                rst.append(pdata)
        return rst
    data = []
    src = '../dataset/chinese-poetry/json/'
    for filename in os.listdir(src):
        if filename.startswith("poet.tang"):
            data.extend(handleJson(src+filename))
    return data




