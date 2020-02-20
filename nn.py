## d1 Train Set All of it
## d2 Test Vector for one
## d3 sonucun pişmemiş hali
# trainVectorNorm train vektorelrının satır bazında vektor norm hesaplamaları (vector(i)=sqrt(s=+xij*xij))

import pandas as pd
import numpy as np

trainVectorNorm = []
d1 = []
trainLabel = []

class knnCosine:
    def __init__(self,d2,testVectorNorm):        
        self.d2 = d2;
        self.rowSize =len(d1)
        self.length = len(d1[0])
        self.vectorTestNorm = testVectorNorm 
        
    def cosine(self,d1,d2,vectorTrainNorm,vectorTestNorm):        
        x = np.dot(d1,d2) 
        if(x == 0):
            return  0
        elif(vectorTrainNorm == 0):
            return 0
        elif(vectorTestNorm == 0):
            return 0
        else:
            return x/(vectorTrainNorm*vectorTestNorm) 
        
    def get(self,k):        
        self.k=k
        d3 = []
        for i in range(self.rowSize):
            d3.append([self.cosine(d1[i],self.d2,trainVectorNorm[i],self.vectorTestNorm),i])                
        # benzerliğe göre sıralama
        d3 = pd.DataFrame(data=d3,columns=['similarity','id'])
        sonuc = pd.DataFrame(d3.sort_values(by=['similarity'],ascending=False))
        sonuc = sonuc.iloc[:k,:]
        for i in range(k):
            sonuc["sinif"] = trainLabel[sonuc.iloc[i,1]]
            
        val  = sonuc.sinif.value_counts().index[0]
        return val
        