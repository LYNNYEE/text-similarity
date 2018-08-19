
import tensorflow as tf
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam,RMSprop
from keras.models import Model,Sequential,load_model
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
from keras import backend as K
import codecs
import LoadData
import numpy as np


max_sen_len = 60
feature_len = 2
wv_len = 300
input_len = wv_len + feature_len

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """
 
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
 
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
 
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
 
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
 
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)
 
    def build(self, input_shape):
        assert len(input_shape) == 3
 
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
 
        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
 
        super(AttentionWithContext, self).build(input_shape)
 
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
 
    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
 
        if self.bias:
            uit += self.b
 
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
 
        a = K.exp(ait)
 
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
 
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
 
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)
 
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

class deep_model:
    def __init__(self,seq_len,wv_size, dtype=tf.float32):
        self.wv_size = wv_size
        self.dtype = dtype

    def CNN_GRU_layer(filter_num,window_size):
        CNN_GRU_net = Sequential()

        # CNN_GRU_net.add(Conv1D(padding="SAME", 
        #           filters=filter_num, 
        #           kernel_size=window_size, 
        #           use_bias=True,
        #           strides=1))
        # CNN_GRU_net.add(BatchNormalization())
        # CNN_GRU_net.add(Activation('relu'))
        # CNN_GRU_net.add(Dropout(0.5))

        # CNN_GRU_net.add(Conv1D(padding="SAME", 
        #           filters=filter_num, 
        #           kernel_size=window_size, 
        #           use_bias=True,
        #           strides=1))

        # CNN_GRU_net.add(BatchNormalization())
        # CNN_GRU_net.add(Activation('relu'))
        # CNN_GRU_net.add(Dropout(0.5))

        CNN_GRU_net.add(Conv1D(padding="SAME", 
                  filters=filter_num, 
                  kernel_size=window_size, 
                  use_bias=True,
                  strides=1))

        #CNN_GRU_net.add(BatchNormalization())
        CNN_GRU_net.add(Activation('relu'))
        CNN_GRU_net.add(Dropout(0.5))
        #CNN_GRU_net.add(GlobalMaxPooling1D())
        #CNN_GRU_net.add(Dense(128, activation='softmax'))

        #带RNN Attention的模型
        # #[b,steps,fileter_num]
        CNN_GRU_net.add(Bidirectional(GRU(max_sen_len, use_bias=True,return_sequences=True),merge_mode='concat'))
        #CNN_GRU_net.add(Bidirectional(GRU(max_sen_len, use_bias=True,return_sequences=True),merge_mode='concat'))
        #CNN_GRU_net.add(AttentionWithContext())
        #CNN_GRU_net.add(Dense(128))
        '''
        CNN_GRU_net.add(Flatten())
        CNN_GRU_net.add(Activation('relu'))
        CNN_GRU_net.add(Dense(128))
        CNN_GRU_net.add(Activation('softmax'))
        CNN_GRU_net.add(BatchNormalization())
        '''
        #CNN_GRU_net.add(Dense(128))
        #CNN_GRU_net.add(Activation('softmax'))
        return CNN_GRU_net




def batched_padding_generator(all_num,batch_size):
    while 1:  
        train_f1 = codecs.open("predata_features/fin_train_tuple1.txt","rb","utf-8")
        train_f2 = codecs.open("predata_features/fin_train_tuple2.txt","rb","utf-8")
        label_f = codecs.open("predata_features/fin_train_label.txt","rb","utf-8")
        line_index = 0
        true_index = 0
        X1 = []
        X2 = []
        L1 = []
        L2 = []
        Y = []
        for line in range(all_num):
            x1 = eval(train_f1.readline())
            x2 = eval(train_f2.readline())
            y = eval(label_f.readline())
            L1.append(len(x1))
            L2.append(len(x2))
            for i in range(max_sen_len-len(x1)):
                x1.append([0 for j in range(input_len)])
            for i in range(max_sen_len-len(x2)):
                x2.append([0 for j in range(input_len)])
            X1.append(x1)
            X2.append(x2)
            Y.append(y)
            line_index +=1
            true_index +=1
            if line_index == batch_size:
                line_index = 0
                #print("generated"+str(np.asarray(X1).shape)+"   "+str(true_index))
                yield ([np.asarray(X1),np.asarray(X2)],Y)
                X1 = []
                X2 = []
                L1 = []
                L2 = []
                Y = []
        train_f1.close()  
        train_f2.close() 
        label_f.close() 

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    #越小越相似时用这个
    distance = y_pred
    #越大越相似时用这个
    #distance = 1 - y_pred
    # weight = 0.75
    # return  K.mean(weight*y_true * K.square(distance) +
    #         (1-weight) * (1 - y_true) * K.square(K.maximum(margin - distance, 0)))
    #越接近，距离越小
    return  K.sum(y_true * K.square(distance) +
         (1 - y_true) * K.square(K.maximum(margin - distance, 0)))
   

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

# 越da越相似
def L2_distance(vects):
    x, y = vects
    #return K.exp(-K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True)))
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    distance = K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
    #return distance
    return K.exp(-distance)
# 越xiao越相似
def L2_distance_low(vects):
    x, y = vects
    #return K.exp(-K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True)))
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    distance = K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
    return distance
    #return K.exp(-distance)
#越大月相似
def cosine_distance(vects):
    x, y = vects
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return K.sum(x * y, axis=-1)
# get batch_size
def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def Mul_attention_shape(shapes):
    shape1, shape2 = shapes
    return [(shape1[0], shape1[1]),(shape2[0], shape2[1])]

#反着的accuracy 看着好看
def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def Mul_attention(rnn_out):
    rnn_out0,rnn_out1 = rnn_out
    rnn_out1 = K.permute_dimensions(rnn_out1,(0,2,1))

    metrix = K.batch_dot(rnn_out0, rnn_out1)
    A2B_row = K.softmax(metrix,axis=1)
    B2A_line = K.softmax(metrix,axis=-1)

    A_weight = K.mean(A2B_row, axis = -1 , keepdims=True)
    B_weight = K.mean(B2A_line, axis = 1 , keepdims=True)

    # out: [b,1,len]*[b,len,?]=[b,1,len]
    A_weight = K.permute_dimensions(A_weight,(0,2,1))
    A = K.batch_dot(A_weight,rnn_out0)
    A = K.squeeze(A, axis=1)
    # out: [b,1,len]*[b,len,?]=[b,1,len]
    rnn_out1 = K.permute_dimensions(rnn_out1,(0,2,1))
    B = K.batch_dot(B_weight,rnn_out1)
    B = K.squeeze(B, axis=1)

    Dense(128)(A)
    Activation('relu')(A)
    #BatchNormalization()(A)

    Dense(128)(B)
    Activation('relu')(B)
    #BatchNormalization()(B)

    return [A,B]


    
def rep_seq(batch_size):
    allNum = 0
    flabel = codecs.open("predata_features/span_train_label.txt","rb","utf-8")
    for line in flabel:
        allNum+=1
    flabel.close()
    print(allNum)
    model = deep_model(max_sen_len,input_len)
    CNN_GRU_layer = deep_model.CNN_GRU_layer(128,3)
    #Xs [batch,seqlen,wvlen]
    text_in1 = Input([max_sen_len,input_len])
    text_out1 = CNN_GRU_layer(text_in1)

    text_in2 = Input([max_sen_len,input_len])
    text_out2 = CNN_GRU_layer(text_in2)

    text_out1,text_out2 =Lambda(Mul_attention,output_shape=Mul_attention_shape)([text_out1,text_out2])
    #正路子！
    distance = Lambda(L2_distance_low,output_shape=eucl_dist_output_shape)([text_out1, text_out2])

    pred_model = Model(inputs= [text_in1,text_in2] , output=distance)
    #checkpoint = ModelCheckpoint('trained_models/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    checkpoint = ModelCheckpoint('trained_models/mul_atten_weight2.hdf5',save_best_only=True)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #rms = RMSprop()
    # pred_model.compile(optimizer=adam, loss=biased_loss, metrics=['accuracy'])
    pred_model.compile(optimizer=adam, loss=contrastive_loss, metrics=[accuracy])
    print("Traning Model...")
    pred_model.fit_generator(batched_padding_generator(allNum,batch_size),validation_data = LoadData.padding_val(), steps_per_epoch = allNum//batch_size, epochs=1000, verbose=1, callbacks=[checkpoint])

            # optimizer = Modified_SGD(
            #     lr=self.learning_rate,
            #     lr_multipliers=learning_rate_multipliers,
            #     momentum=0.5)


def predict():
    model = load_model("trained_models/mul_atten_weight2.hdf5",custom_objects={'BatchNormalization':BatchNormalization,'contrastive_loss':contrastive_loss})
    #model = load_model("trained_models/mul_atten_weight.hdf5",custom_objects={'AttentionWithContext':AttentionWithContext,'contrastive_loss':contrastive_loss})
    pre_X = LoadData.padding_submit()
    #pre_X = [[pre_X[0][0],pre_X[0][1],pre_X[0][2],pre_X[0][3]],[pre_X[1][0],pre_X[1][1],pre_X[1][2],pre_X[1][3]]]
    prediction = model.predict(pre_X)
    fin_f = codecs.open("result.txt","wb","utf-8")
    fintext = ""
    for p in prediction:
        #越小越相似时用这个
        
        n = p[0]
        if n>1:
            n=1
        n = 1-n
        
        #越大月相似时用这个
        #n = p[0]
        fintext+=str(n)+"\n"
    fin_f.write(fintext)
    fin_f.close()
    print(prediction)


#rep_seq(400)
predict()

        



    # def Conv_layer(X,window_size,width,filter_num):
    #     X = tf.expand_dims(X,-1)
    #     W_conv = tf.Variable(tf.truncated_normal([window_size,width, 1,1],stddev = 0.1))
    #     b_conv = tf.Variable(tf.truncated_normal([1],stddev = 0.1))
    #     conv = tf.nn.conv2d(
    #             X,
    #             W_conv,
    #             [1,1,1,1],
    #             "SAME",
    #             use_cudnn_on_gpu=False,
    #             data_format='NHWC',
    #         )
    #     conv = tf.nn.relu(conv + b_conv)
    #     return conv

    # def CNN():
    #     conv1 = self.Conv_layer(X,window_size,self.wv_size,out_channel)
    #     drop = tf.nn.dropout(conv1,self.keep_prob)