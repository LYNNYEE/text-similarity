import re
import nltk
import numpy as np
import sys 
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer('spanish')
snowball_stemmer.stem
from collections import Iterable
from nltk.corpus import stopwords
stops = set(stopwords.words("spanish"))

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer ,TfidfTransformer
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer
import difflib
from fuzzywuzzy import fuzz
import networkx as nx

class Features():
    def __init__(self,train,test):

        self.stptrain_data = [[],[]]
        self.entrain_data = [[],[]]
        self.test_data = [[],[]]
        for index, row in train.iterrows():
            q1 = nltk.word_tokenize(row.spanish1)
            q2 = nltk.word_tokenize(row.spanish2)
            self.stptrain_data[0].append(" ".join(q1))
            self.stptrain_data[1].append(" ".join(q2))
            q3 = nltk.word_tokenize(row.english1)
            q4 = nltk.word_tokenize(row.english2)
            self.entrain_data[0].append(" ".join(q3))
            self.entrain_data[1].append(" ".join(q4))

        for index, row in test.iterrows():
            q1 = nltk.word_tokenize(row.spanish1)
            q2 = nltk.word_tokenize(row.spanish2)
            self.test_data[0].append(" ".join(q1))
            self.test_data[1].append(" ".join(q2))

        # self.train_vecs = [[[[] for w in s.split(" ")] for s in t] for t in self.train_data]
        # self.test_vecs = [[[[] for w in s.split(" ")] for s in t] for t in self.test_data]

    def document_tfidf(self):
        esdocs = self.stptrain_data[0]+self.stptrain_data[1]+self.test_data[0]+self.test_data[1]
        estfidf_vectorizer = TfidfVectorizer()
        eslsa = TruncatedSVD(n_components=150)
        transformed = estfidf_vectorizer.fit_transform(esdocs)
        esdoc_tfidf = eslsa.fit_transform(transformed)
        estrain_doc_tfidf = esdoc_tfidf[0:len(self.stptrain_data[0])*2]
        estest_doc_tfidf = esdoc_tfidf[len(self.stptrain_data[0])*2:]
        normalizer = Normalizer(copy=False)
        estrain_doc_tfidf = normalizer.fit_transform(estrain_doc_tfidf)
        normalizer = Normalizer(copy=False)
        estest_doc_tfidf = normalizer.fit_transform(estest_doc_tfidf)

        endocs = self.entrain_data[0]+self.entrain_data[1]
        entfidf_vectorizer = TfidfVectorizer()
        enlsa = TruncatedSVD(n_components=150)
        transformed = entfidf_vectorizer.fit_transform(endocs)
        endoc_tfidf = eslsa.fit_transform(transformed)
        normalizer = Normalizer(copy=False)
        endoc_tfidf = normalizer.fit_transform(endoc_tfidf)

        train_pair1 = estrain_doc_tfidf[::2]
        train_pair2 = estrain_doc_tfidf[1::2]

        submit_pair1 = estest_doc_tfidf[::2]
        submit_pair2 = estest_doc_tfidf[1::2]

        en_trian_pair1 = endoc_tfidf[::2]
        en_trian_pair2 = endoc_tfidf[1::2]

        es_train_sim,submit_sim,en_trian_sim = [],[],[]

        for i in range(len(train_pair1)):
            es_train_sim.append(cosine_similarity([train_pair1[i],train_pair2[i]])[0][1])
        for i in range(len(submit_pair1)):
            submit_sim.append(cosine_similarity([submit_pair1[i],submit_pair2[i]])[0][1])
        for i in range(len(en_trian_pair1)):
            en_trian_sim.append(cosine_similarity([en_trian_pair1[i],en_trian_pair2[i]])[0][1])

        return [estrain_doc_tfidf,estest_doc_tfidf,endoc_tfidf],[es_train_sim,submit_sim,en_trian_sim]

    def inlist(wlist,qlist):
        for qw in qlist:
            if qw in wlist:
                return 1
        return 0

    #是否都包含否定词
    def contains_nagtive(self):
        spnegw = ['no','nadie','nada','nunca','jamás','ningún','ninguno','ninguna','ningunos','ningunas']
        engnegw = ['no','never','little', 'few','not','can\'t','none','isn\'t','hardly','seldom','neither','nothing','nobody','nor']
        stptrain_ratio = []
        stpsubmit_ratio = []
        entrain_ratio = []
        for i in range(len(self.stptrain_data[0])):
            q1 = Features.inlist(spnegw,self.stptrain_data[0][i].split(" "))
            q2 = Features.inlist(spnegw,self.stptrain_data[1][i].split(" "))
            stptrain_ratio.append([[q1,q2],int(q1==q2)])
        for i in range(len(self.test_data[0])):
            q1 = Features.inlist(spnegw,self.test_data[0][i].split(" "))
            q2 = Features.inlist(spnegw,self.test_data[1][i].split(" "))
            stpsubmit_ratio.append([[q1,q2],int(q1==q2)])
        for i in range(len(self.entrain_data[0])):
            q1 = Features.inlist(engnegw,self.entrain_data[0][i].split(" "))
            q2 = Features.inlist(engnegw,self.entrain_data[1][i].split(" "))
            entrain_ratio.append([[q1,q2],int(q1==q2)])
        return [stptrain_ratio,stpsubmit_ratio,entrain_ratio]

    def fuzzy(self):
        stptrain_ratio = []
        stpsubmit_ratio = []
        entrain_ratio = []
        for i in range(len(self.stptrain_data[0])):
            q1 = self.stptrain_data[0][i]
            q2 = self.stptrain_data[1][i]
            stptrain_ratio.append([fuzz.ratio(q1,q2),fuzz.partial_ratio(q1,q2),fuzz.token_sort_ratio(q1,q2),fuzz.token_set_ratio(q1,q2)])
        for i in range(len(self.test_data[0])):
            q1 = self.test_data[0][i]
            q2 = self.test_data[1][i]
            stpsubmit_ratio.append([fuzz.ratio(q1,q2),fuzz.partial_ratio(q1,q2),fuzz.token_sort_ratio(q1,q2),fuzz.token_set_ratio(q1,q2)])
        for i in range(len(self.entrain_data[0])):
            q1 = self.entrain_data[0][i]
            q2 = self.entrain_data[1][i]
            entrain_ratio.append([fuzz.ratio(q1,q2),fuzz.partial_ratio(q1,q2),fuzz.token_sort_ratio(q1,q2),fuzz.token_set_ratio(q1,q2)])
        return [stptrain_ratio,stpsubmit_ratio,entrain_ratio]

    #是否应该除以总数？
    def q_freq(self):
        esdocs = self.stptrain_data[0]+self.stptrain_data[1]+self.test_data[0]+self.test_data[1]
        endocs = self.entrain_data[0]+self.entrain_data[1]
        #freq_dict
        es_freq = {}
        en_freq = {}
        for s in set(esdocs):
            es_freq[s] = esdocs.count(s)
        for s in set(endocs):
            en_freq[s] = endocs.count(s)
        #freq
        stptrain_ratio = []
        stpsubmit_ratio = []
        entrain_ratio = []
        for i in range(len(self.stptrain_data[0])):
            q1 = es_freq[self.stptrain_data[0][i]]
            q2 = es_freq[self.stptrain_data[1][i]]
            stptrain_ratio.append([[q1,q2],abs(q1-q2)])
        for i in range(len(self.test_data[0])):
            q1 = es_freq[self.test_data[0][i]]
            q2 = es_freq[self.test_data[1][i]]
            stpsubmit_ratio.append([[q1,q2],abs(q1-q2)])
        for i in range(len(self.entrain_data[0])):
            q1 = en_freq[self.entrain_data[0][i]]
            q2 = en_freq[self.entrain_data[1][i]]
            entrain_ratio.append([[q1,q2],abs(q1-q2)])
        return [stptrain_ratio,stpsubmit_ratio,entrain_ratio]

    def pagerank(rows1,rows2):
        qid_graph = {}
        for (row1,row2) in zip(rows1,rows2):
            qid_graph.setdefault(row1, []).append(row2)
            qid_graph.setdefault(row2, []).append(row1)

        MAX_ITER = 40
        d = 0.85  
        pagerank_dict = {i:1/len(qid_graph) for i in qid_graph}
        num_nodes = len(pagerank_dict)
        for iter in range(0, MAX_ITER):
            for node in qid_graph:    
                local_pr = 0
                for neighbor in qid_graph[node]:
                    local_pr += pagerank_dict[neighbor]/len(qid_graph[neighbor])
                pagerank_dict[node] = (1-d)/num_nodes + d*local_pr
        return pagerank_dict

    def q_pagerank(self):
        esdocs0 = self.stptrain_data[0]+self.test_data[0]
        esdocs1 = self.stptrain_data[1]+self.test_data[1]
        endocs0 = self.entrain_data[0]
        endocs1 = self.entrain_data[1]
        esrank = Features.pagerank(esdocs0,esdocs1)
        enrank = Features.pagerank(endocs0,endocs1)
        stptrain_ratio = []
        stpsubmit_ratio = []
        entrain_ratio = []
        for i in range(len(self.stptrain_data[0])):
            q1 = esrank[self.stptrain_data[0][i]]
            q2 = esrank[self.stptrain_data[1][i]]
            stptrain_ratio.append([[q1,q2],abs(q1-q2)])
        for i in range(len(self.test_data[0])):
            q1 = esrank[self.test_data[0][i]]
            q2 = esrank[self.test_data[1][i]]
            stpsubmit_ratio.append([[q1,q2],abs(q1-q2)])
        for i in range(len(self.entrain_data[0])):
            q1 = enrank[self.entrain_data[0][i]]
            q2 = enrank[self.entrain_data[1][i]]
            entrain_ratio.append([[q1,q2],abs(q1-q2)])
        return [stptrain_ratio,stpsubmit_ratio,entrain_ratio]

    def q_len(self):
        stptrain_ratio = []
        stpsubmit_ratio = []
        entrain_ratio = []
        for i in range(len(self.stptrain_data[0])):
            q1 = len(self.stptrain_data[0][i].replace(" ",""))
            q2 = len(self.stptrain_data[1][i].replace(" ",""))
            stptrain_ratio.append([[q1,q2],abs(q1-q2)])
        for i in range(len(self.test_data[0])):
            q1 = len(self.test_data[0][i].replace(" ",""))
            q2 = len(self.test_data[1][i].replace(" ",""))
            stpsubmit_ratio.append([[q1,q2],abs(q1-q2)])
        for i in range(len(self.entrain_data[0])):
            q1 = len(self.entrain_data[0][i].replace(" ",""))
            q2 = len(self.entrain_data[1][i].replace(" ",""))
            entrain_ratio.append([[q1,q2],abs(q1-q2)])
        return [stptrain_ratio,stpsubmit_ratio,entrain_ratio]

    def difflib_ratios(self):
        stptrain_ratio = []
        stpsubmit_ratio = []
        entrain_ratio = []   
        for i in range(len(self.stptrain_data[0])):
            seq = difflib.SequenceMatcher()
            seq.set_seqs(self.stptrain_data[0][i],self.stptrain_data[1][i])
            stptrain_ratio.append(seq.ratio())
        for i in range(len(self.test_data[0])):
            seq = difflib.SequenceMatcher()
            seq.set_seqs(self.test_data[0][i],self.test_data[1][i])
            stpsubmit_ratio.append(seq.ratio())
        for i in range(len(self.entrain_data[0])):
            seq = difflib.SequenceMatcher()
            seq.set_seqs(self.entrain_data[0][i],self.entrain_data[1][i])
            entrain_ratio.append(seq.ratio())
        return [stptrain_ratio,stpsubmit_ratio,entrain_ratio]

    def flatten(items, ignore_types=(str, bytes)):
        for x in items:
            if isinstance(x, Iterable) and not isinstance(x, ignore_types):
                yield from Features.flatten(x)
            else:
                yield x

    def gen_features(self):
        f6 = self.q_pagerank()
        f7 = self.difflib_ratios()
        _,f1 = self.document_tfidf()
        f2 =  self.contains_nagtive()
        f3 = self.fuzzy()
        f4 = self.q_freq()
        f5 = self.q_len()

        estrain = []
        estest = []
        entrain = []

        for i in range(len(f1[0])):
            vec = list(Features.flatten([f1[0][i]]+[f2[0][i][1]]+[f3[0][i]]+[f4[0][i][1]]+[f5[0][i][1]]+[f6[0][i][1]]+[f7[0][i]]))
            pairs = list(Features.flatten([f2[0][i][0]]+[f4[0][i][0]]+[f5[0][i][0]]+[f6[0][i][0]]))
            #vec = list(Features.flatten([f1[0][i]]+[f5[0][i]]))
            estrain.append([vec,pairs])
        for i in range(len(f1[1])):
            vec = list(Features.flatten([f1[1][i]]+[f2[1][i][1]]+[f3[1][i]]+[f4[1][i][1]]+[f5[1][i][1]]+[f6[1][i][1]]+[f7[1][i]]))
            pairs = list(Features.flatten([f2[1][i][0]]+[f4[1][i][0]]+[f5[1][i][0]]+[f6[1][i][0]]))
            estest.append([vec,pairs])
        for i in range(len(f1[2])):
            vec = list(Features.flatten([f1[2][i]]+[f2[2][i][1]]+[f3[2][i]]+[f4[2][i][1]]+[f5[2][i][1]]+[f6[2][i][1]]+[f7[2][i]]))
            pairs = list(Features.flatten([f2[2][i][0]]+[f4[2][i][0]]+[f5[2][i][0]]+[f6[2][i][0]]))
            entrain.append([vec,pairs])
        return estrain,estest,entrain


#否定词   [是否可以考虑成，该词是否是否定词？]
'''
def negativeWordsCount(row):
    q1 = nltk.word_tokenize(row['spanish1'])
    q2 = nltk.word_tokenize(row['spanish2'])

    Ncount1 = 0
    Ncount2 = 0
    negw = ['no','nadie','nada','nunca','jamás','ningún','ninguno','ninguna','ningunos','ningunas']
    for wd in negw:
        Ncount1 += q1.count(wd)
        Ncount2 += q2.count(wd)
    
    fs = list()
        fs.append(Ncount1)
        fs.append(Ncount2)

    if Ncount1 == Ncount2:
        if Ncount1 == 0:
                    fs.append(0.)
            else:
                    fs.append(1.)
    else:
        if Ncount1 == 0 or Ncount2 == 0:
            fs.append(2.)
        else:
            fs.append(3.)

    return fs
'''
'''
def format_data(stptrain,stptest):
    train_data = [[],[]]
    test_data = [[],[]]
    for index, row in stptrain.iterrows():
        q1 = nltk.word_tokenize(row.spanish1)
        q2 = nltk.word_tokenize(row.spanish2)
        train_data[0].append(" ".join(q1))
        train_data[1].append(" ".join(q2))
    for index, row in stptest.iterrows():
        q1 = nltk.word_tokenize(row.spanish1)
        q2 = nltk.word_tokenize(row.spanish2)
        test_data[0].append(" ".join(q1))
        test_data[1].append(" ".join(q2))
    return train_data,test_data

def tfidf(train_data,test_data,train_vecs,test_vecs):
    train_num = len(train_data[0])
    test_num = len(test_data[0])
    print((train_num,test_num))
    vectorizer = CountVectorizer() 
    corpus = train_data[0]+train_data[1]+test_data[0]+test_data[1]
    X = vectorizer.fit_transform(corpus)
    words = vectorizer.get_feature_names()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X).toarray()
    print(tfidf.shape)

    print(" train tfidf model over")
    words = vectorizer.get_feature_names()
    for ti,tra in enumerate(train_data):
        for si,sen in enumerate(tra):
            for wi,w in enumerate(sen.split(" ")):
                if w in words:
                    train_vecs[ti][si][wi].append(tfidf[train_num*ti+si,words.index(w)])
                else:
                    train_vecs[ti][si][wi].append(0)
    for ti,tra in enumerate(test_data):
        for si,sen in enumerate(tra):
            for wi,w in enumerate(sen.split(" ")):
                if w in words:
                    test_vecs[ti][si][wi].append(tfidf[train_num*2+test_num*ti+si,words.index(w)])
                else:
                    test_vecs[ti][si][wi].append(0)

def spanish_nagtiveWords(train_data,test_data,train_vecs,test_vecs):
    negw = ['no','nadie','nada','nunca','jamás','ningún','ninguno','ninguna','ningunos','ningunas']
    for ti,tra in enumerate(train_data):
        for si,sen in enumerate(tra):
            for wi,w in enumerate(sen.split(" ")):
                if w in negw:
                    train_vecs[ti][si][wi].append(1)
                else:
                    train_vecs[ti][si][wi].append(0)
    for ti,tra in enumerate(test_data):
        for si,sen in enumerate(tra):
            for wi,w in enumerate(sen.split(" ")):
                if w in negw:
                    test_vecs[ti][si][wi].append(1)
                else:
                    test_vecs[ti][si][wi].append(0)

def generate_wordfeatrue(train_data,test_data):
    train_data,test_data = format_data(train_data,test_data)
    train_vecs = [[[[] for w in s.split(" ")] for s in t] for t in train_data]
    test_vecs = [[[[] for w in s.split(" ")] for s in t] for t in test_data]
    tfidf(train_data,test_data,train_vecs,test_vecs)
    spanish_nagtiveWords(train_data,test_data,train_vecs,test_vecs)
    return train_vecs,test_vecs
'''