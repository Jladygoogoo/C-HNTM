'''
参考代码：https://github.com/awslabs/w-lda
'''
import os
import re
import pickle
from tqdm import tqdm
import pandas as pd
from gensim.corpora.mmcorpus import MmCorpus
from gensim.corpora import Dictionary
import nltk
nltk.data.path.append("D:/program_files/nltk_data")
from nltk.tokenize import word_tokenize
import spacy
nlp = spacy.load("en_core_web_sm")


import utils
from vectorize import PretrainBERT


def my_tokenize(text, stopwords):
    words = []
    for word in word_tokenize(text.lower()):
        if len(word) < 2:
            continue
        if word not in stopwords and not re.search(r'=|\'|-|`|\.|<|>|[0-9]|_|-|~|\^|\*|\\|\||\+', word):
            words.append(word)
    return words



def is_document_start(line):
    '''
    Parse wiki103 *.tokens original files
    '''
    if len(line) < 4:
        return False
    if line[0] == '=' and line[-1] == '=':
        if line[2] != '=':
            return True
        else:
            return False
    else:
        return False


def token_list_per_doc(input_dir, token_file):
    '''
    Parse wiki103 *.tokens original files
    '''
    lines_list = []
    line_prev = ''
    prev_line_start_doc = False
    with open(os.path.join(input_dir, token_file), 'r', encoding='utf-8') as f:
        for l in f:
            line = l.strip()
            if prev_line_start_doc and line:
                # the previous line should not have been start of a document!
                lines_list.pop()
                lines_list[-1] = lines_list[-1] + ' ' + line_prev

            if line:
                if is_document_start(line) and not line_prev:
                    lines_list.append(line)
                    prev_line_start_doc = True
                else:
                    lines_list[-1] = lines_list[-1] + ' ' + line
                    prev_line_start_doc = False
            else:
                prev_line_start_doc = False
            line_prev = line

    print("{} documents parsed!".format(len(lines_list)))
    return lines_list    



def parse_raw_docs_20news(data_dir):
    '''
    Parse 20news original files from directory
    '''
    raw_docs = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if "DS" in file: continue
            try:
                doc = open(os.path.join(root, file), encoding='utf-8').read()
            except:
                doc = open(os.path.join(root, file), encoding='cp1252').read()
            lines = doc.splitlines()
            new_lines = []
            for i in range(lines.index(""), len(lines)):
                if "@" in lines[i] and "write" in lines[i]:
                    continue
                if "in article" in lines[i].lower():
                    continue
                new_lines.append(lines[i])
            doc = '\n'.join(new_lines)
            raw_docs.append(doc)
    return raw_docs


def parse_ag_news_docs(filepath):
    '''
    Parse docs from MXM *.csv file
    param: filepath: *.csv file, contains titles and descriptions. Extract descriptions as docs
    return: docs, list of str
    '''
    df = pd.read_csv(filepath)
    docs = list(df["description"].values)
    return docs


def parse_mxm_bows(filepath):
    '''
    Parse bows from MXM *.txt file
    param: filepath: *.txt file, contains bows information
    return: bows, list of (int,int)
    '''
    data = open(filepath, encoding='utf8').read().splitlines()
    bows = []
    for line in data:
        flag = line.find(',')
        flag = line.find(',', flag+1)
        line = line[flag+1:]
        bow = list(eval("{{ {} }}".format(line)).items())
        bow = list(map(lambda p:(p[0]-1,p[1]), bow))
        bows.append(bow)
    return bows



def preprocess_and_save_from_raw_docs(raw_train_docs, raw_test_docs, keep_n, base_dir, data_source_name):
    '''
    Generate and save dictionary, bows, tokenized docs, vecs based on raw docs
    param: raw_train_docs: list of str
    param: raw_test_docs: list of str
    param: keep_n: for dictionary.filter_extreme()
    param: base_dir: save dirpath
    param: data_source_name 
    '''
    # tokenize
    stopwords = open("../data/stopwords.txt").read().splitlines()
    utils.print_log("Tokenizing docs...")
    train_docs = []
    for doc in raw_train_docs:
        doc = my_tokenize(doc, stopwords)
        if len(doc) > 10:
            train_docs.append(doc)    
    test_docs = []
    for doc in raw_test_docs:
        doc = my_tokenize(doc, stopwords)
        if len(doc) > 10:
            test_docs.append(doc)
    docs = train_docs + test_docs
    utils.print_log("total docs size: {} | train docs size: {} | test docs size: {}".format(
        len(docs), len(train_docs), len(test_docs)))

    # dictionary
    utils.print_log("Preparing dictionary...")
    dictionary = Dictionary(docs)
    ori_dict_size = len(dictionary)
    dictionary.filter_extremes(keep_n=keep_n, no_above=1, no_below=1)
    utils.print_log("original dictionary size: {} | filtered dictionary size: {} (keep_n={})".format(
        ori_dict_size, len(dictionary), keep_n
    ))
    dictionary.id2token = {v:k for k,v in dictionary.token2id.items()}

    # bows
    utils.print_log("Preparing bows...")
    train_bows = [dictionary.doc2bow(doc) for doc in train_docs]
    test_bows = [dictionary.doc2bow(doc) for doc in test_docs]

    # vectorize
    utils.print_log("Start vectorizing...")
    embedding_model = PretrainBERT()
    vecs = []
    for word in tqdm(dictionary.token2id.keys()):
        vecs.append(embedding_model.get_embedding(word))

    # save
    save_path = os.path.join(base_dir, "{}_keep-{}_dictionary.pkl".format(data_source_name, keep_n))
    with open(save_path, 'wb') as f:
        pickle.dump(dictionary, f)
    utils.print_log("Dictionary saved: {}".format(save_path))

    save_path = os.path.join(base_dir, "{}_keep-{}_vecs.pkl".format(data_source_name, keep_n))
    with open(save_path, 'wb') as f:
        pickle.dump(vecs, f)
    utils.print_log("Vectors saved: {}".format(save_path))    

    train_save_path = os.path.join(base_dir, "{}_keep-{}_docs_train.txt".format(data_source_name, keep_n))
    test_save_path = os.path.join(base_dir, "{}_keep-{}_docs_test.txt".format(data_source_name, keep_n))
    with open(train_save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join([' '.join(doc) for doc in train_docs]))
    with open(test_save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join([' '.join(doc) for doc in test_docs]))
    utils.print_log("Docs saved: {}, {}".format(train_save_path, test_save_path))

    train_save_path = os.path.join(base_dir, "{}_keep-{}_bows_train.mm".format(data_source_name, keep_n))
    test_save_path = os.path.join(base_dir, "{}_keep-{}_bows_test.mm".format(data_source_name, keep_n))
    MmCorpus.serialize(train_save_path, train_bows)
    MmCorpus.serialize(test_save_path, test_bows)
    utils.print_log("Bows saved: {}, {}".format(train_save_path, test_save_path))




def preprocess_20news():
    data_source_name = "20news"    
    base_dir = "../data/20news"
    keep_n = 2000  

    # raw docs
    raw_train_dir = "../data/20news/raw/20news-bydate/20news-bydate-train/"
    raw_test_dir = "../data/20news/raw/20news-bydate/20news-bydate-test/"
    raw_train_docs = parse_raw_docs_20news(raw_train_dir)
    raw_test_docs = parse_raw_docs_20news(raw_test_dir)

    preprocess_and_save_from_raw_docs(raw_train_docs, raw_test_docs, keep_n, base_dir, data_source_name)



def preprocess_wiki103():
    '''
    Preprocess Wiki-103 data
    '''
    data_source_name = "wiki103"    
    base_dir = "../data/wiki103"
    input_dir = os.path.join(base_dir, "raw/wikitext-103")
    keep_n = 20000
    
    # raw docs
    train_file = 'wiki.train.tokens'
    val_file = 'wiki.valid.tokens'
    test_file = 'wiki.test.tokens'
    raw_train_docs = token_list_per_doc(input_dir, train_file)
    raw_test_docs = token_list_per_doc(input_dir, test_file) + token_list_per_doc(input_dir, val_file)

    preprocess_and_save_from_raw_docs(raw_train_docs, raw_test_docs, keep_n, base_dir, data_source_name)


def preprocess_agnews():
    '''
    Preprocess AG News
    data source: "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
    '''
    data_source_name = "ag_news"
    base_dir = "../data/ag_news"
    keep_n = 10000
    train_file = "../data/ag_news/raw/train.csv"
    test_file = "../data/ag_news/raw/test.csv"

    raw_train_docs = parse_ag_news_docs(train_file)
    raw_test_docs = parse_ag_news_docs(test_file)

    preprocess_and_save_from_raw_docs(raw_train_docs, raw_test_docs, keep_n, base_dir, data_source_name)




def preprocess_mxm():
    '''
    Preprocess Million Song Dataset lyrics
    '''
    data_source_name = "mxm"    
    base_dir = "../data/mxm"
    vocab_file = "../data/mxm/raw/vocab.txt"
    raw_train_file = "../data/mxm/raw/mxm_dataset_train.txt"
    raw_test_file = "../data/mxm/raw/mxm_dataset_test.txt"

    vocab = open(vocab_file, encoding='utf8').read().splitlines()
    stopwords = open("../data/stopwords.txt").read().splitlines()
    keep_idx = []
    for i, word in enumerate(vocab):
        if word not in stopwords:
            keep_idx.append(i)
    utils.print_log("original vocab size: {} | filtered vocab size: {}".format(
        len(vocab), len(keep_idx)
    ))    

    utils.print_log("Generating new bows from old bows...")
    old_train_bows = parse_mxm_bows(raw_train_file)
    old_test_bows = parse_mxm_bows(raw_test_file)
    old_bows = old_train_bows + old_test_bows
    new_docs = []
    for bow in old_bows:
        doc = []
        for idx, count in bow:
            if idx in keep_idx:
                doc.extend([vocab[idx] for _ in range(count)])
        new_docs.append(doc)
    
    utils.print_log("Preparing dictionary...")
    dictionary = Dictionary(new_docs)
    dictionary.id2token = {v:k for k,v in dictionary.token2id.items()}
    vocab_old_new_idx_map = {}
    for w in dictionary.token2id:
        vocab_old_new_idx_map[vocab.index(w)] = dictionary.token2id[w]

    new_train_bows = []
    for bow in old_train_bows:
        new_train_bows.append([(vocab_old_new_idx_map[p[0]], p[1]) for p in bow if p[0] in keep_idx])
    new_test_bows = []
    for bow in old_test_bows:
        new_test_bows.append([(vocab_old_new_idx_map[p[0]], p[1]) for p in bow if p[0] in keep_idx])

    # vectorize
    utils.print_log("Start vectorizing...")
    embedding_model = PretrainBERT()
    vecs = []
    for word in tqdm(dictionary.token2id.keys()):
        vecs.append(embedding_model.get_embedding(word))

    # save
    save_path = os.path.join(base_dir, "{}_dictionary.pkl".format(data_source_name))
    with open(save_path, 'wb') as f:
        pickle.dump(dictionary, f)
    utils.print_log("Dictionary saved: {}".format(save_path))

    save_path = os.path.join(base_dir, "{}_vecs.pkl".format(data_source_name))
    with open(save_path, 'wb') as f:
        pickle.dump(vecs, f)
    utils.print_log("Vectors saved: {}".format(save_path))    

    train_save_path = os.path.join(base_dir, "{}_bows_train.mm".format(data_source_name))
    test_save_path = os.path.join(base_dir, "{}_bows_test.mm".format(data_source_name))
    MmCorpus.serialize(train_save_path, new_train_bows)
    MmCorpus.serialize(test_save_path, new_test_bows)
    utils.print_log("Bows saved: {}, {}".format(train_save_path, test_save_path))    
    
    

def preprocess_reuters():
    # raw_docs = reuters.paras()
    return


def preprocess_rcv1():
    # base_dir = "../data/rcv1"
    # raw_data_dir = os.path.join(base_dir, "raw/rcv1")
    # NO_BELOW = 100 # 词频过滤
    # SOURCE_NAME = "rcv1"

    # ids = []
    # raw_docs = []
    # topic_classes = []
    # # 70万数据太难顶了，先拿前5万看看
    # count = 0
    # max_count = 50000
    # for root, dirs, files in os.walk(raw_data_dir):
    #     for file in files:
    #         if "xml" not in file: continue
    #         filepath = os.path.join(root, file)
    #         content = open(filepath).read()
    #         root_ele = et.XML(content)
    #         ids.append(root_ele.attrib["itemid"])
    #         for child in root_ele:
    #             # if child.tag == "title":
    #             #     print("title:", child.text)
    #             if child.tag == "text":
    #                 text = ""
    #                 for node in child:
    #                     text += '\n' + node.text
    #                 raw_docs.append(text)
    #             if child.tag == "metadata":
    #                 topics = []
    #                 for node in child:
    #                     if "class" not in node.attrib: continue
    #                     if "topic" in node.attrib["class"]:
    #                         for code_node in node:
    #                             topics.append(code_node.attrib["code"])
    #                 topic_classes.append(topics)
    #         count += 1
    #         if count >= max_count: break
    #     if count >= max_count: break    
    return



if __name__ == '__main__':
    preprocess_wiki103()
    # preprocess_20news()
    # preprocess_mxm()
    # preprocess_agnews()
