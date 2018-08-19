import model6
import codecs
import LoadData
max_sen_len=70
input_len = 300
#pretrained = 'trained_models/transfer/model/weight.17.hdf5'
pretrained = 'trained_models/transfer/model/weight.hdf5'
save_path = 'trained_models/transfer/model'

def train():
    for i in range(4):
        if i != 3:
        	continue
        print("--------------------"+str(i)+"------------------------")
        #model6.rep：seq(128,"es",str(i),save_path+str(i)+"/weight",pretrained)
        model6.rep_seq(128,"es",str(i),save_path+str(i)+"/weight",pretrained)

def predict():
    pre_X = model6.padding_submit()
    Y = [[] for i in range(4)] 
    for i in range(4):
        print("--------------------"+str(i)+"------------------------")
        pred = model6.predict_stack("trained_models/transfer/model"+str(i)+"/weight.hdf5",pre_X)
        for p in pred:
            Y[i].append(p[0])
    fin = []
    fintext = ""
    fin_f = codecs.open("result.txt","wb","utf-8")
    for i in range(10000):
        y = sum([Y[j][i] for j in range(4)])/4
        fintext+=str(y)+"\n"
    fin_f.write(fintext)
    fin_f.close()
        

#model6.rep_seq(128,"en","",save_path+"/weight")
#LoadData.rep()
#train()
predict()