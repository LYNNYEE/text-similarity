
import tensorflow as tf
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam,RMSprop
from keras.models import Model,Sequential,load_model
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional,TimeDistributed
from keras import backend as K
from keras.activations import softmax
from keras.utils import *
import tensorflow as tf
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam,RMSprop
from keras.models import Model,Sequential,load_model
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional,TimeDistributed
from keras import backend as K
from keras.activations import softmax
from keras.utils import *
import codecs
import LoadData
import numpy as np
from keras.callbacks import Callback,EarlyStopping
from sklearn.metrics import f1_score, precision_score, recall_score
import Fmeasure
from keras import regularizers

max_sen_len = 70
feature_len = 2
wv_len = 300
#input_len = wv_len + feature_len
input_len = wv_len

# class Metrics(Callback):
#     def on_train_begin(self, logs={}):
#         self.val_f1s = []
#         self.val_recalls = []
#         self.val_precisions = []

#     def on_epoch_end(self, epoch, logs={}):
#         #此处需要取数值
#         val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
#         val_targ = self.validation_data[1]
#         _val_f1 = f1_score(val_targ, val_predict)
#         _val_recall = recall_score(val_targ, val_predict)
#         _val_precision = precision_score(val_targ, val_predict)
#         self.val_f1s.append(_val_f1)
#         self.val_recalls.append(_val_recall)
#         self.val_precisions.append(_val_precision)
#         return



##############读数据用##############
def padding_val(lang,model_index):
    test_f1 = codecs.open("predata_features/data"+str(model_index)+"/"+lang+"val0.txt","rb","utf-8")
    test_f2 = codecs.open("predata_features/data"+str(model_index)+"/"+lang+"val1.txt","rb","utf-8")
    label_f = codecs.open("predata_features/data"+str(model_index)+"/"+lang+"val_label.txt","rb","utf-8")
    feature_vec = codecs.open("predata_features/data"+str(model_index)+"/"+lang+"val_feature.txt","rb","utf-8")
    test_pairs = [[],[]]
    test_features = []
    test_features_pairs = []
    test_label = []
    for tl in test_f1:
        line = eval(tl)
        for i in range(max_sen_len-len(line)):
            line.append([0 for j in range(input_len)])
        test_pairs[0].append(line)
    for tl in test_f2:
        line = eval(tl)
        for i in range(max_sen_len-len(line)):
            line.append([0 for j in range(input_len)])
        test_pairs[1].append(line)
    for tl in label_f:
        y=eval(tl)
        test_label.append(y)

    for tl in feature_vec:
        line = eval(tl)
        test_features.append(line[0])
        test_features_pairs.append(line[1])

    #print("test dataset load over")
    return test_pairs+[test_features]+[test_features_pairs],test_label
    #return test_pairs+[test_features],test_label

def padding_submit():
    test_f1 = codecs.open("predata_features/span_submit_pairs1.txt","rb","utf-8")
    test_f2 = codecs.open("predata_features/span_submit_pairs1.txt","rb","utf-8")
    feature_vec = codecs.open("predata_features/submit_feature.txt","rb","utf-8")
    test_pairs = [[],[]]
    test_features = []
    test_features_pairs = []
    for tl in test_f1:
        line = eval(tl)
        for i in range(max_sen_len-len(line)):
            line.append([0 for j in range(input_len)])
        test_pairs[0].append(line)
    for tl in test_f2:
        line = eval(tl)
        for i in range(max_sen_len-len(line)):
            line.append([0 for j in range(input_len)])
        test_pairs[1].append(line)

    for tl in feature_vec:
        line = eval(tl)
        test_features.append(line[0])
        test_features_pairs.append(line[1])

    print("test dataset load over")
    print(len(test_pairs[0]))
    return test_pairs+[test_features]+[test_features_pairs]

def batched_padding_generator(all_num,batch_size,lang,model_index):
    while 1:  
        train_f1 = codecs.open("predata_features/data"+str(model_index)+"/"+lang+"train0.txt","rb","utf-8")
        train_f2 = codecs.open("predata_features/data"+str(model_index)+"/"+lang+"train1.txt","rb","utf-8")
        label_f = codecs.open("predata_features/data"+str(model_index)+"/"+lang+"train_label.txt","rb","utf-8")
        feature_vec = codecs.open("predata_features/data"+str(model_index)+"/"+lang+"train_feature.txt","rb","utf-8")
        
        line_index,true_index = 0,0
        X1,X2,F,FP,L1,L2,Y = [],[],[],[],[],[],[]
        for line in range(all_num):
            x1 = eval(train_f1.readline())
            x2 = eval(train_f2.readline())
            y = eval(label_f.readline())
            t = eval(feature_vec.readline())
            f = t[0]
            fp = t[1]
            L1.append(len(x1))
            L2.append(len(x2))
            for i in range(max_sen_len-len(x1)):
                x1.append([0 for j in range(input_len)])
            for i in range(max_sen_len-len(x2)):
                x2.append([0 for j in range(input_len)])
            X1.append(x1)
            X2.append(x2)
            F.append(f)
            FP.append(fp)
            Y.append(y)
            
            line_index +=1
            true_index +=1
            if line_index == batch_size:
                line_index = 0
                #yield ([np.asarray(X1),np.asarray(X2),np.asarray(F)],Y)
                yield ([np.asarray(X1),np.asarray(X2),np.asarray(F),np.asarray(FP)],Y)
                X1,X2,F,FP,L1,L2,Y = [],[],[],[],[],[],[]
        train_f1.close()  
        train_f2.close() 
        label_f.close() 
##############读数据用################
def CNN_GRU_layer(filter_num,window_size):
    CNN_GRU_net = Sequential()
    CNN_GRU_net.add(Masking(mask_value=0., input_shape=(max_sen_len, wv_len)))
    # CNN_GRU_net.add(LSTM(150, use_bias=True,return_sequences=True))
    # CNN_GRU_net.add(LSTM(150, use_bias=True,return_sequences=True))
    CNN_GRU_net.add(Bidirectional(GRU(150, use_bias=True,return_sequences=True),merge_mode='concat'))
    CNN_GRU_net.add(Bidirectional(GRU(150, use_bias=True,return_sequences=True),merge_mode='concat'))
    CNN_GRU_net.add(TimeDistributed(Dense(150,activation="tanh")))
    return CNN_GRU_net

def out_layer():
    outnet = Sequential()
    outnet.add(Dense(300, activation='elu'))
    outnet.add(BatchNormalization())
    outnet.add(Dropout(0.5))
    outnet.add(Dense(300, activation='elu'))
    outnet.add(BatchNormalization())
    outnet.add(Dropout(0.5))
    return outnet

class Mul_Attention(Layer):
    def __init__(self, *args, **kwargs):
        self.supports_masking = True
        super(Mul_Attention, self).__init__(*args, **kwargs)
    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None
    #def build(self, input_shape):
    def call(self, x, mask=None):
        #compute masking
        mask0 = K.cast(mask[0],dtype='float32')
        mask1 = K.cast(mask[1],dtype='float32')
        mask_nums0 = K.sum(mask0,axis=-1)
        mask_nums0 = K.repeat(K.expand_dims(mask_nums0),150)
        mask_nums0 = K.reshape(mask_nums0,(-1,150))
        mask_nums1 = K.sum(mask1,axis=-1)
        mask_nums1 = K.repeat(K.expand_dims(mask_nums1),150)
        mask_nums1 = K.reshape(mask_nums1,(-1,150))

        mask0 = K.repeat(mask0,150)
        mask0 = K.permute_dimensions(mask0,(0,2,1))
        mask1 = K.repeat(mask1,150)
        mask1 = K.permute_dimensions(mask1,(0,2,1))

        rnn_out0,rnn_out1 = x

        mask_out0 = Multiply()([rnn_out0,mask0])
        mask_out1 = Multiply()([rnn_out1,mask1])
        mask_out1 = K.permute_dimensions(mask_out1,(0,2,1))
        metrix = Lambda(lambda x: tf.einsum('aij,ajk->aik',x[0],x[1]))([mask_out0,mask_out1])
        A_weight = K.softmax(metrix,axis = 1)
        B_weight = K.softmax(metrix,axis = -1)
        alpha = K.mean(A_weight, axis = -1 , keepdims=True)
        beta = K.mean(B_weight, axis = 1 , keepdims=True)
        #[TA,1]
        gamma_A = Lambda(lambda x: tf.einsum('aij,ajk->aik',x[0],x[1]))([beta,K.permute_dimensions(B_weight,(0,2,1))])  #[1,n]
        gamma_B = Lambda(lambda x: tf.einsum('aij,ajk->aik',x[0],x[1]))([K.permute_dimensions(alpha,(0,2,1)),A_weight]) #[1,m]
        #[C,1]
        A = Lambda(lambda x: tf.einsum('aij,ajk->aik',x[0],x[1]))([gamma_A,rnn_out0])
        B = Lambda(lambda x: tf.einsum('aij,ajk->aik',x[0],x[1]))([gamma_B,rnn_out1])
        # A = K.permute_dimensions(A,(0,2,1))
        # B = K.permute_dimensions(B,(0,2,1))
        A = K.batch_flatten(A)
        B = K.batch_flatten(B)

        A = A/mask_nums0
        B = B/mask_nums1
        return [A,B,alpha,beta]
    def compute_output_shape(self, input_shape):
        shape1, shape2 = input_shape
        return [(shape1[0],shape1[2]),(shape2[0],shape2[2]),(shape1[1],),(shape2[1],)]



def rep_seq(batch_size,lang,model_index,save_path,pretrained = None):
    allNum = 0
    flabel = codecs.open("predata_features/data"+str(model_index)+"/"+lang+"train0.txt","rb","utf-8")
    for line in flabel:
        allNum+=1
    flabel.close()
    CNN_GRU = CNN_GRU_layer(128,3)
    #Xs [batch,seqlen,wvlen]
    text_in1 = Input([max_sen_len,input_len])
    text_out1 = CNN_GRU(text_in1)
    text_in2 = Input([max_sen_len,input_len])
    text_out2 = CNN_GRU(text_in2)

    text_out1,text_out2,alpha,beta =Mul_Attention()([text_out1,text_out2])

    #加 乘
    diff = Lambda(lambda x: K.abs(x[0] - x[1]))([text_out1,text_out2])
    mul = Lambda(lambda x: x[0] * x[1])([text_out1,text_out2])

    feature_in = Input([10,])
    feature_dense = BatchNormalization()(feature_in)
    feature_dense = Dense(64, activation='relu')(feature_dense)

    feature_pair_in = Input([8,])
    feature_pair_dense = BatchNormalization()(feature_pair_in)
    feature_pair_dense = Dense(64, activation='relu')(feature_pair_dense)

    outnet = out_layer()

    textout = Concatenate(axis=-1)([diff,mul,feature_dense,feature_pair_dense])
    #textout = Concatenate(axis=-1)([text_out1,text_out2,feature_dense])
    textout = BatchNormalization()(textout)
    #textout = Dropout(0.4)(textout)
    #textout = Dense(1,activation="sigmoid")(textout)
    textout = outnet(textout)
    textout = Dense(1,activation="sigmoid")(textout)

    # 可视化
    # get_4rd_layer_output = K.function([pred_model.layers[0].input,pred_model.layers[1].input,pred_model.layers[3].input],[pred_model.layers[4].output])
    # layer_output = get_4rd_layer_output(padding_val())[0]
    # print(layer_output)
    if pretrained is None:
        pred_model = Model(inputs= [text_in1,text_in2,feature_in,feature_pair_in] , outputs=textout)
    else:
        pred_model = load_model(pretrained,custom_objects={'Mul_Attention':Mul_Attention,'BatchNormalization':BatchNormalization})
    
    class_weight = {0: 1.309028344, 1: 0.472001959}

    print(pred_model.summary())
    #checkpoint = ModelCheckpoint('trained_models/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    checkpoint = ModelCheckpoint(save_path+'.{epoch:02d}.hdf5',save_best_only=False)
    earlystop = EarlyStopping(monitor='val_loss', patience=7, verbose=0, mode='auto')
    # def as_keras_metric(method):
    #     import functools
    #     from keras import backend as K
    #     import tensorflow as tf
    #     @functools.wraps(method)
    #     def wrapper(self, args, **kwargs):
    #         """ Wrapper for turning tensorflow metrics into keras metrics """
    #         value, update_op = method(self, args, **kwargs)
    #         K.get_session().run(tf.local_variables_initializer())
    #         with tf.control_dependencies([update_op]):
    #             value = tf.identity(value)
    #         return value
    #     return wrapper
    # precision = as_keras_metric(tf.metrics.precision)
    # recall = as_keras_metric(tf.metrics.recall)
    pred_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #pred_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras_metrics.precision(), keras_metrics.recall()])
    print("Traning Model...")
    #metrics = Metrics()
    pred_model.fit_generator(batched_padding_generator(allNum,batch_size,lang,model_index),class_weight=class_weight,validation_data = padding_val(lang,model_index), steps_per_epoch = allNum//batch_size, epochs=35, verbose=1, callbacks=[checkpoint,earlystop])
    #pred_model.fit_generator(batched_padding_generator(allNum,batch_size),validation_data = padding_val(), steps_per_epoch = allNum//batch_size, epochs=1000, verbose=1, callbacks=[checkpoint])


def predict():
    model = load_model("trained_models/transfer/feature_weight_sigmoid_es1.08.hdf5",custom_objects={'Mul_Attention':Mul_Attention,'BatchNormalization':BatchNormalization})
    #model = load_model("trained_models/mul_atten_weight.hdf5",custom_objects={'AttentionWithContext':AttentionWithContext,'contrastive_loss':contrastive_loss})
    #pre_X = padding_val()[0]
    pre_X = padding_submit()
    prediction = model.predict(pre_X)
    fin_f = codecs.open("result.txt","wb","utf-8")
    fintext = ""
    for p in prediction:
        '''
        #越小越相似时用这个
        n = p[0]
        if n>1:
            n=1
        n = 1-n
        '''
        #越大月相似时用这个
        n = p[1]
        #n = p
        fintext+=str(n)+"\n"
    fin_f.write(fintext)
    fin_f.close()
    print(prediction)

def predict_stack(path,pre_X):
    model = load_model(path,custom_objects={'Mul_Attention':Mul_Attention,'BatchNormalization':BatchNormalization})
    prediction = model.predict(pre_X)
    return prediction

#rep_seq(400,"en","")
#predict()















# def recall(y_true, y_pred):
#     """Recall metric.
#     Only computes a batch-wise average of recall.
#     Computes the recall, a metric for multi-label classification of
#     how many relevant items are selected.
#     """
#     # x0 = K.sum(K.equal(y_true,K.cast(K.round(y_pred), y_true.dtype))) / K.sum(K.equal(1.0,K.cast(y_true, y_true.dtype)))
#     # x1 = K.sum(K.equal(y_true,K.cast(K.round(y_pred), y_true.dtype))) / K.sum(K.equal(0.0,K.cast(y_true, y_true.dtype)))

#     x0 = K.cast(K.equal(y_true[0],K.round(y_pred[0])), y_true.dtype) #(Pred=True|True)
#     x1 = K.cast(K.equal(y_true[1],K.round(y_pred[1])), y_true.dtype)
#     y0 = K.sum(y_true[0])
#     y1 = K.sum(y_true[1])

#     recall = K.sum([x0,x1])
#     return recall