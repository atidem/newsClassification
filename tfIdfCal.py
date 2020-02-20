## Bu bölümde Frekans , Tfidf , Rowsum , VectorNorm hesapları yapılmaktadır.

import pandas as pd
import numpy as np
from TurkishStemmer import TurkishStemmer ##elasticSearch
from math import sqrt

kok = TurkishStemmer()
count = []
vectorNorm = []

class getFrequency:
    def __init__(self,spor,saglik,magazin,ekonomi):
        self.spor = spor
        self.saglik = saglik
        self.magazin = magazin
        self.ekonomi = ekonomi
        count.clear()

## parça parça gelen dataframler tek tek frekansı hesaplanıp birleştiriliyor.  
    def get(self,features):
        self.features = features 
        data = self.update(self.ekonomi,self.features)
        ekonomiFreq = pd.DataFrame(data=data,columns=features)
        ekonomiFreq["nameOfTxt"] = self.ekonomi.iloc[:,0]
        ekonomiFreq["classOfTxt"] = "ekonomi"
        
        data = self.update(self.saglik,self.features)
        saglikFreq = pd.DataFrame(data=data,columns=features)
        saglikFreq["nameOfTxt"] = self.saglik.iloc[:,0]
        saglikFreq["classOfTxt"] = "saglik"
        
        data = self.update(self.spor,self.features)
        sporFreq = pd.DataFrame(data=data,columns=features)
        sporFreq["nameOfTxt"] = self.spor.iloc[:,0]
        sporFreq["classOfTxt"] = "spor"
        
        data = self.update(self.magazin,self.features)
        magazinFreq = pd.DataFrame(data=data,columns=features)
        magazinFreq["nameOfTxt"] = self.magazin.iloc[:,0]
        magazinFreq["classOfTxt"] = "magazin"
        
        frames = [ekonomiFreq,saglikFreq,sporFreq,magazinFreq]
        docFreq = pd.concat(frames,ignore_index=True)
        
        return docFreq
    
## frekans hesaplıyor bunu yaparken de satır toplamlarını alıyoruz tf*idf de kullanmak için 
## try catch kullanıldı dataframedeki her satrıda farklı sayıda kelıme bulundugu için
    def update(self,data,features = []):
        row = len(data.iloc[:,1]) 
        matrix = np.zeros(row*len(features)).reshape(row,len(features))
        data = data.values
        for i in range(len(data)):
            c = 0
            for j in range(1,len(data[1])):
                try:
                    np.isnan(data[i][j])
                except TypeError:
                    try:
                        string = str(data[i][j])
                        num = features.index(string)
                        matrix[i][num] = matrix[i][num]+1
                        c = c + 1
                    except ValueError:
                        pass
                    
            count.append(c)              
        return matrix  
    
## vektör normları ve tf idf burada hesaplanıyor malum sqrt hız problemi
## .values kullanımı dataframe i numpy array e cevırıyor daha hızlı gezinebilmek için
class tdIdf:
    def __init__(self,frequency,length,feature):
        self.frequency = frequency        
        self.length = length       
        self.feature = feature
        self.columnSize =len(self.feature)
        self.liste = frequency.ix[:,:self.columnSize].values
        
    def calc(self):
        vectorNorm.clear()
        
        for i in range(self.length):
            vector = 0
            for j in range(1,len(self.liste[1])-1):
                x = self.liste[i][j]
                if(x != 0):                                       
                    self.liste[i][j] = (x/count[i])*np.log10(self.length/x)
                    vector = vector + (self.liste[i][j]*self.liste[i][j])
            vectorNorm.append(sqrt(vector))
        return self.liste
                    
