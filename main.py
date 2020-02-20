## bu kod txtlerinizi kendince düzenler ve yeni dosya oluşturarak işlemlerine devam eder..
## turkishstemmer kullanıldı.
## kütüphaneleri "pip install kütüphane adı" kodu ile cmd üzerinden indirebilirsiniz..
## protected_words dosyasını aşagıdaki konuma kopyalayınız..
## C:\Users\kullanıcı adı\Anaconda3\Lib\site-packages\TurkishStemmer\resources

import pandas as pd
import numpy as np
import textNormalize as nm
import tfIdfCal  as tf
import nn as nn
import time
from TurkishStemmer import TurkishStemmer ##elasticSearch


#############   regex , tokenize  , stopwords işlemleri ########################
t = time.process_time()
kok = TurkishStemmer()
tfIdfTest = pd.DataFrame()
tfIdfTrain = pd.DataFrame()
testData = 320
trainData = 600
classes = ["ekonomi","magazin","saglik","spor"]

nm.changeMarks("train\\ekonomi")
nm.applyStopWords("train\\ekonomi")
trainEkonomi = nm.getDataFrames("train\\ekonomi").get()

nm.changeMarks("train\\magazin")
nm.applyStopWords("train\\magazin")
trainMagazin = nm.getDataFrames("train\\magazin").get()

nm.changeMarks("train\\spor")
nm.applyStopWords("train\\spor")
trainSpor = nm.getDataFrames("train\\spor").get()

nm.changeMarks("train\\saglik")
nm.applyStopWords("train\\saglik")
trainSaglik = nm.getDataFrames("train\\saglik").get()
##################################################################################

############## Tekrarsiz feature oluşturma #######################################
tmp = []
tmp.append("nameOfTxt")
features = tmp + list(set(nm.features))
columnSize = len(features)
#################################################################################

#############  Train Tf*Idf İşlemi ##############################################
dataFreqTrain = tf.getFrequency(trainSpor,trainSaglik,trainMagazin,trainEkonomi).get(features)
tfIdfTrain = tf.tdIdf(dataFreqTrain,trainData,features).calc()
tfIdfTrain = pd.DataFrame(data=tfIdfTrain,columns=features)
tfIdfTrain["nameOfTxt"] = dataFreqTrain["nameOfTxt"]
tfIdfTrain["classOfTxt"] = dataFreqTrain["classOfTxt"]
trainVectorNorm = tf.vectorNorm.copy()

elapsed_time = time.process_time() - t
print ("train verileri işlendi",elapsed_time)
##################################################################################

#############    regex , tokenize  , stopwords işlemleri ########################
t = time.process_time()
    
nm.changeMarks("test\\spor")
nm.applyStopWords("test\\spor")
testSpor = nm.getDataFrames("test\\spor").get()

nm.changeMarks("test\\ekonomi")
nm.applyStopWords("test\\ekonomi")
testEkonomi = nm.getDataFrames("test\\ekonomi").get()

nm.changeMarks("test\\saglik")
nm.applyStopWords("test\\saglik")
testSaglik = nm.getDataFrames("test\\saglik").get()

nm.changeMarks("test\\magazin")
nm.applyStopWords("test\\magazin")
testMagazin = nm.getDataFrames("test\\magazin").get()
##################################################################################

############## Test verileri Tf*Idf Hesapları ####################################
dataFreqTest = tf.getFrequency(testSpor,testSaglik,testMagazin,testEkonomi).get(features)
tfIdfTest = tf.tdIdf(dataFreqTest,testData,features).calc()
tfIdfTest = pd.DataFrame(data=tfIdfTest,columns=features)
tfIdfTest["nameOfTxt"] = dataFreqTest["nameOfTxt"]
tfIdfTest["classOfTxt"] = dataFreqTest["classOfTxt"]
testVectorNorm = tf.vectorNorm.copy()

elapsed_time = time.process_time() - t
print ("test verileri işlendi",elapsed_time)
###################################################################################

############## Matrisleri Uygun Şekilde bölünür ##################################
t = time.process_time()

trainx = tfIdfTrain.iloc[:,1:columnSize].values
trainy = tfIdfTrain.iloc[:,columnSize]
testx = tfIdfTest.iloc[:,1:columnSize].values
testy = tfIdfTest.iloc[:,columnSize]
predict = []

elapsed_time = time.process_time() - t
print ("matrisler bölündü.",elapsed_time)
###################################################################################

"""
############ K-NN  ###############################################################
t = time.process_time()
print("ve knn baslar..")
predict.clear()
nn.d1 = trainx
nn.trainVectorNorm = trainVectorNorm
nn.trainLabel = trainy 
k = 3 ## k değeri değiştirilebilir.
for i in range(320):
    a = nn.knnCosine(testx[i],testVectorNorm[i]).get(k)
    predict.append(a)
               
elapsed_time = time.process_time() - t
print ("knn tamamlandı.",elapsed_time)
###################################################################################
""" 

from sklearn.svm import SVC
svc = SVC(kernel="linear")
svc.fit(trainx,trainy)

predict = svc.predict(testx)


########### Presicion Recall Fscore ###############################################
sonuc = pd.DataFrame()
sonuc["tahmin"] = predict
sonuc["gercek"] = testy

measure = ["presicion","recall","fscore"]
calc = []
calc.clear()
for i in classes:
    tp = 0
    fp = 0
    fn = 0
    recall = 0
    presicion = 0
    fscore = 0
    for j in range(len(predict)):
        if(testy[j] == i or predict[j] == i):
            if(predict[j] == testy[j]):
                tp = tp + 1
            elif(predict[j] == i and testy[j] != i ):
                fp = fp + 1
            else:
                fn = fn + 1
    recall = tp/(tp+fn)
    presicion = tp/(tp+fp)
    fscore = (2* presicion * recall) / (presicion + recall)
    calc.append([presicion,recall,fscore])
    

calc.append(list(np.average(calc,axis=0)))
labels = classes.copy()
labels.append("average")
calc = pd.DataFrame(data=calc,index=labels,columns = measure)
#################1###################################################################

######### Dosyaya yazdırma işlmleri #################################################
df = pd.concat([tfIdfTest,tfIdfTrain],ignore_index=True)     
df.to_csv(r'tf-idf.txt',index=False)
calc.to_csv(r'measure svm.txt',index=True)                
##################################################################################### 
     
######### Sunum İçin ################################################################
table = pd.read_csv("measure k=5.txt")
#####################################################################################
