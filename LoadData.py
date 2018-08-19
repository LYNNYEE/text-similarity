#coding:utf-8
import io
import sys 
#sys.setdefaultencoding('utf-8')
import numpy as np
import pandas as pd
import os
import codecs
import re
import pickle
import nltk
from nltk.stem import SnowballStemmer
spsnowball_stemmer = SnowballStemmer('spanish')
ensnowball_stemmer = SnowballStemmer('english')
#snowball_stemmer.stem
from nltk.corpus import stopwords
spstops = set(stopwords.words("spanish"))
enstops = set(stopwords.words("english"))
from keras.utils import *
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.cross_validation import cross_val_score
import random
from features import *

#训练集最大长度是42 测试集最大56

max_sen_len = 70
feature_len = 2
wv_len = 300
#input_len = wv_len + feature_len
input_len = wv_len

def subchar(text):
	text=text.replace("á", "a")
	text=text.replace("ó", "o")
	text=text.replace("é", "e")	
	text=text.replace("í", "i")
	text=text.replace("ú", "u")	
	return text

def cleanSpanish(df):
	#处理反符号
	df['spanish1'] = df.spanish1.map(lambda x: re.sub("¿"," ¿ ",x))
	df['spanish2'] = df.spanish2.map(lambda x: re.sub("¿"," ¿ ",x))
	df['spanish1'] = df.spanish1.map(lambda x: ' '.join([spsnowball_stemmer.stem(word) for word in nltk.word_tokenize(x.lower().encode('utf-8').decode('utf-8'))]).encode('utf-8'))
	df['spanish2'] = df.spanish2.map(lambda x: ' '.join([spsnowball_stemmer.stem(word) for word in nltk.word_tokenize(x.lower().encode('utf-8').decode('utf-8'))]).encode('utf-8'))
	if 'english1' in df.keys():
		df['english1'] = df.english1.map(lambda x: ' '.join([ensnowball_stemmer.stem(word) for word in nltk.word_tokenize(x.lower().encode('utf-8').decode('utf-8'))]).encode('utf-8'))
		df['english2'] = df.english2.map(lambda x: ' '.join([ensnowball_stemmer.stem(word) for word in nltk.word_tokenize(x.lower().encode('utf-8').decode('utf-8'))]).encode('utf-8'))
	#西班牙语缩写还原#

def removestopwords(df, spstop,enstop):
	# df['spanish1'] = df.spanish1.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8'))  if word not in spstop]))
	# df['spanish2'] = df.spanish2.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8'))  if word not in spstop]))
	# if 'english1' in df.keys():
	# 	df['english1'] = df.english1.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8'))  if word not in enstop]))
	# 	df['english2'] = df.english2.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8'))  if word not in enstop]))
	#不去停用词 挺好的
	df['spanish1'] = df.spanish1.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8')) ]))
	df['spanish2'] = df.spanish2.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8')) ]))
	if 'english1' in df.keys():
		df['english1'] = df.english1.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8')) ]))
		df['english2'] = df.english2.map(lambda x: ' '.join([word for word in nltk.word_tokenize(x.decode('utf-8')) ]))


def isnum(s):
	try:
		float(s)
		return True
	except ValueError:
		pass
	try:
		import unicodedata
		unicodedata.numeric(s)
		return True
	except (TypeError, ValueError):
		pass
	return False
def loadvec(fn):
	embeddings_index = {}
	f = io.open(fn,encoding = 'utf8')
	for line in f:
		values = line.split()
		word = values[0]
		i = 1
		for ch in values[1:]:
			if isnum(ch) is True:
					break
			else:
				word += (' '+values[i])
				i += 1
		coefs = np.asarray(values[i:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	return embeddings_index

def record(strs,path):
	file = codecs.open(path,'wb','utf-8')
	[file.write(str(line)+"\n") for line in strs] 
	file.close()

#保留 处理oov
def readvec(wv,w,ov):
	if w in wv.keys():
		#print(len(list(wv[w])))
		return list(wv[w])
	else:
		#print(w)
		if w not in ov.keys():
			ov[w] = [random.uniform(-0.5,0.5) for i in range(wv_len)]
		return ov[w]
def rep():
	print('===read data===')
	df_train_en_sp = pd.read_csv('data/cikm_english_train_20180516.txt',sep='	', header=None,error_bad_lines=False)
	df_train_sp_en = pd.read_csv('data/cikm_spanish_train_20180516.txt',sep='	', header=None,error_bad_lines=False)
	df_train_en_sp.columns = ['english1', 'spanish1', 'english2', 'spanish2', 'result']
	df_train_sp_en.columns = ['spanish1', 'english1', 'english2', 'spanish2', 'result']

	test_data = pd.read_csv('data/cikm_test_b_20180730.txt', sep='	', header=None,error_bad_lines=False)
	test_data.columns = ['spanish1', 'spanish2']

	print("===clean text===")
	cleanSpanish(df_train_en_sp)
	cleanSpanish(df_train_sp_en)
	# train_data1 = df_train_en_sp.drop(['english1','english2'],axis=1)
	# train_data2 = df_train_sp_en.drop(['english1','english2'],axis=1)
	train_data = pd.concat([df_train_en_sp,df_train_sp_en],axis=0)

	print(test_data.shape)
	cleanSpanish(test_data)
	stptrain = train_data.copy()
	stptest = test_data.copy()

	removestopwords(stptrain, spstops,enstops)
	removestopwords(stptest, spstops,enstops)

	span_train_pairs = ([],[])
	span_train_label = []
	en_train_pairs = ([],[])
	en_train_label = []
	span_test_pairs = ([],[])
	esov = {}
	enov = {}

	####count features#### 
	
	#train_features ,test_features = features.generate_wordfeatrue(stptrain,stptest)
	
	features = Features(stptrain,stptest)
	estrain,estest,entrain = features.gen_features()
	slice_num = len(estrain)//4
	'''
	for i in range(4):
		val = estrain[i*slice_num:(i+1)*slice_num]
		train = estrain[0:i*slice_num]+estrain[(i+1)*slice_num:]
		record(train,"predata_features/data"+str(i)+"/estrain_feature.txt")
		record(val,"predata_features/data"+str(i)+"/esval_feature.txt")
	record(estest,"predata_features/submit_feature.txt")
	record(entrain[6000:],"predata_features/entrain_feature.txt")
	record(entrain[0:6000],"predata_features/enval_feature.txt")
	'''
	
	
	#mix
	# mix_train = np.asarray(list(map(lambda x, y: [x,y], estrain, entrain))).reshape(-1,8)
	# file = codecs.open("predata_features/estrain_featurevec.txt",'wb','utf-8')
	# [file.write(str(list(line))+"\n") for line in mix_train[6000:]] 
	# file.close()
	# file = codecs.open("predata_features/esval_featurevec.txt",'wb','utf-8')
	# [file.write(str(list(line))+"\n") for line in mix_train[0:6000]] 
	# file.close()
	# file = codecs.open("predata_features/essubmit_featurevec.txt",'wb','utf-8')
	# [file.write(str(list(line))+"\n") for line in estest] 
	# file.close()
	
	# spanWv = loadvec("data/wiki.es.vec")
	spanWv = loadvec("data/wiki.multi.es.vec")
	enWv = loadvec("data/wiki.multi.en.vec")

	double_pairs = []
	
	#录入词向量
	for _, row in stptrain.iterrows():
		q1 = nltk.word_tokenize(row.spanish1)
		q2 = nltk.word_tokenize(row.spanish2)
		q3 = nltk.word_tokenize(row.english1)
		q4 = nltk.word_tokenize(row.english2)
		#q1 = [readvec(spanWv,w,esov) + train_features[0][index][wi] for wi,w in enumerate(q1)]
		q1 = [readvec(spanWv,w,esov) for wi,w in enumerate(q1)]
		q2 = [readvec(spanWv,w,esov) for wi,w in enumerate(q2)]
		q3 = [readvec(enWv,w,enov) for wi,w in enumerate(q3)]
		q4 = [readvec(enWv,w,enov) for wi,w in enumerate(q4)]

		span_train_pairs[0].append(q1)
		span_train_pairs[1].append(q2)

		if row.result == 1:
			double_pairs.append((q1,q2))

		en_train_pairs[0].append(q3)
		en_train_pairs[1].append(q4)

		span_train_label.append(row.result)
		en_train_label.append(row.result)

	'''
	for double in double_pairs:
		i = random.randint(0,20000)
		span_train_pairs[0].insert(i,double[0])
		span_train_pairs[1].insert(i,double[1])
		span_train_label.insert(i,"1")
	'''
	
	for _, row in stptest.iterrows():
		q1 = nltk.word_tokenize(row.spanish1)
		q2 = nltk.word_tokenize(row.spanish2)
		span_test_pairs[0].append([readvec(spanWv,w,esov)  for wi,w in enumerate(q1)] )
		span_test_pairs[1].append([readvec(spanWv,w,esov)  for wi,w in enumerate(q2)] )
	
	print("start_recoding words")

	slice_num = len(span_train_pairs[0])//4
	for i in range(4):
		val0 = span_train_pairs[0][i*slice_num:(i+1)*slice_num]
		val1 = span_train_pairs[1][i*slice_num:(i+1)*slice_num]
		train0 = span_train_pairs[0][0:i*slice_num]+span_train_pairs[0][(i+1)*slice_num:]
		train1 = span_train_pairs[1][0:i*slice_num]+span_train_pairs[1][(i+1)*slice_num:]
		vallabel = span_train_label[i*slice_num:(i+1)*slice_num]
		trainlabel = span_train_label[0:i*slice_num]+span_train_label[(i+1)*slice_num:]

		val_feature = estrain[i*slice_num:(i+1)*slice_num]
		train_feature = estrain[0:i*slice_num]+estrain[(i+1)*slice_num:]

		double_pairs = []

		for di,l in enumerate(trainlabel):
			if l == 1:
				double_pairs.append((train0[di],train1[di],train_feature[di]))

		for d in double_pairs:
			di = random.randint(0,slice_num*3)
			train0[di].append(d[0])
			train1[di].append(d[1])
			trainlabel[di].append(1)
			train_feature.append(di[2])

		double_pairs = []

		for di,l in enumerate(vallabel):
			if l == 1:
				double_pairs.append((val0[di],val1[di],val_feature[di]))

		for d in double_pairs:
			di = random.randint(0,slice_num)
			val0[di].append(d[0])
			val1[di].append(d[1])
			vallbel[di].append(1)
			val_feature.append(di[2])



		record(train0,"predata_features/data"+str(i)+"/estrain0.txt")
		record(train1,"predata_features/data"+str(i)+"/estrain1.txt")
		record(trainlabel,"predata_features/data"+str(i)+"/estrain_label.txt")

		record(val0,"predata_features/data"+str(i)+"/esval0.txt")
		record(val1,"predata_features/data"+str(i)+"/esval1.txt")
		record(vallabel,"predata_features/data"+str(i)+"/esval_label.txt")

		record(train_feature,"predata_features/data"+str(i)+"/estrain_feature.txt")
		record(val_feature,"predata_features/data"+str(i)+"/esval_feature.txt")


	record(estest,"predata_features/submit_feature.txt")
	record(entrain[6000:],"predata_features/entrain_feature.txt")
	record(entrain[0:6000],"predata_features/enval_feature.txt")

	record(span_test_pairs[0],"predata_features/span_submit_pairs1.txt")
	record(span_test_pairs[1],"predata_features/span_submit_pairs2.txt")
	
	# file = codecs.open("predata_features/span_submit_pairs1.txt",'wb','utf-8')
	# [file.write(str(list(line))+"\n") for line in span_test_pairs[0]] 
	# file.close()

	# file = codecs.open("predata_features/span_submit_pairs2.txt",'wb','utf-8')
	# [file.write(str(list(line))+"\n") for line in span_test_pairs[1]] 
	# file.close()

# def padding_val():
# 	test_f1 = codecs.open("predata_features/mix_test_pairs1.txt","rb","utf-8")
# 	test_f2 = codecs.open("predata_features/mix_test_pairs2.txt","rb","utf-8")
# 	label_f = codecs.open("predata_features/mix_test_label.txt","rb","utf-8")
# 	test_pairs = [[],[]]
# 	test_label = []
# 	for tl in test_f1:
# 		line = eval(tl)
# 		for i in range(max_sen_len-len(line)):
# 			line.append([0 for j in range(input_len)])
# 		test_pairs[0].append(line)
# 	for tl in test_f2:
# 		line = eval(tl)
# 		for i in range(max_sen_len-len(line)):
# 			line.append([0 for j in range(input_len)])
# 		test_pairs[1].append(line)
# 	for tl in label_f:
# 		test_label.append(eval(tl))

# 	print("test dataset load over")
# 	return test_pairs,np_utils.to_categorical(test_label)
# 	#return test_pairs,test_label

# def padding_submit():
# 	test_f1 = codecs.open("predata_features/span_submit_pairs1.txt","rb","utf-8")
# 	test_f2 = codecs.open("predata_features/span_submit_pairs2.txt","rb","utf-8")
# 	test_pairs = [[],[]]
# 	test_label = []
# 	for tl in test_f1:
# 		line = eval(tl)
# 		for i in range(max_sen_len-len(line)):
# 			line.append([0 for j in range(input_len)])
# 		test_pairs[0].append(line)
# 	for tl in test_f2:
# 		line = eval(tl)
# 		for i in range(max_sen_len-len(line)):
# 			line.append([0 for j in range(input_len)])
# 		test_pairs[1].append(line)

# 	print("test dataset load over")
# 	return test_pairs
	# batch_data_1 = []
	# batch_data_2 = []
	# batch_label = []
	# for i in range(int(len(span_train_label)/batch_size)):
	# 	batch_data_1.append(span_train_pairs[0][i*batch_size:(i+1)*batch_size])
	# 	batch_data_2.append(span_train_pairs[1][i*batch_size:(i+1)*batch_size])
	# 	batch_label.append(span_train_label[i*batch_size:(i+1)*batch_size])
	# return batch_data_1,batch_data_2,batch_label




#rep()
# padding()












	










