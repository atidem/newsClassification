
##  !! Attention Please Hilal station.. !!
## bu kod txtlerinizi kendince düzenler ve yeni dosya oluşturarak işlemlerine devam eder..
## protected words dosyasını aşagıdaki konuma kopyalayınız..
## C:\Users\kullanıcı adı\Anaconda3\Lib\site-packages\TurkishStemmer\resources
## turkishstemmer kullanıldı.
## kütüphaneleri "pip install kütüphane adı" kodu ile cmd üzeinden indirebilirsiniz..

from TurkishStemmer import TurkishStemmer ##elasticSearch
import pandas as pd
import glob

features = []
kok = TurkishStemmer()

##kütüphaneden gelen stemmer
## klasördeki tüm *.txt ler için noktalama işaretleri kaldırma işlemi
class changeMarks:
    def __init__ (self,path):
        self.path=path
        pathGeneral = path+'\\*.txt'
        files = glob.glob(pathGeneral)
        for name in files:
            with open(name,"r") as f:
                data = f.read()
                f.close()
                data = data.replace("'"," ")
                data = data.replace('"'," ")
                data = data.replace("."," ")
                data = data.replace(","," ")
                data = data.replace(":"," ")
                data = data.replace(";"," ")
                data = data.replace("("," ")
                data = data.replace(")"," ")
                data = data.replace("?"," ")
                data = data.replace("!"," ")
                data = data.replace("-"," ")
                data = data.replace("..."," ")
                data = data.replace("<"," ")
                data = data.replace(">"," ")
                data = data.replace("%"," ")
                data = data.replace("&"," ")
                data = data.replace("$"," ")
                data = data.replace("+"," ")
                data = data.replace("{"," ")
                data = data.replace("}"," ")
                data = data.replace("["," ")
                data = data.replace("]"," ")
                data = data.replace("’"," ")
                data = data.replace("‘"," ")
                data = data.replace("“"," ")
                data = data.replace("”"," ")
                data = data.replace("*"," ")
                data = data.replace("/"," ")
                data = data.replace('\\',' ' )
                data = data.replace("0"," ")
                data = data.replace("1"," ")
                data = data.replace("2"," ")
                data = data.replace("3"," ")
                data = data.replace("4"," ")
                data = data.replace("5"," ")
                data = data.replace("6"," ")
                data = data.replace("7"," ")
                data = data.replace("8"," ")
                data = data.replace("9"," ")
                
                name=str(name).replace(" ","")
                change = open(name+"a","w+")
                change.write(data)
                change.close()
            
## txt to dataframe    
class getDataFrames:
    def __init__ (self,path):
        self.path=path
        
    def getStems(self,data=[]):   
        for i in range(1,len(data)):
            data[i] = kok.stem(data[i].lower())
        return data
    
    def get(self):
        pathGeneral = self.path+'\\*.txta'
        files = glob.glob(pathGeneral)
        data = []
        for name in files:
                with open(name,"r") as f:
                    data.append(self.getStems(f.read().split())) 
                    f.close()
        data = pd.DataFrame(data=data)
        data.fillna(value=pd.np.nan, inplace=True)
        return data


## stop_words dosyasını diziye aldık.
class getStopWords:
    def __init__():
        file = open("stop_words.txt","r")
        data = file.read().split()
        file.close()
        return data

## stopwords u uyguluyoruz , her seferinde stop words u gezmek daha maliyetli
## verilerdeki kelime sayısı daha az oldugundan verilerdeki kelimeler stopwords de varsa siliyor.
## bu karşılaştırmayı kelimeleri küçülterek yapıyor.
## yoksa exceptiondan yakalayıp geri sektirip kökleyip stringe ekleyip geri yazıyoruz.
class applyStopWords(getStopWords):
    def __init__ (self,path):
        self.path=path        
        data = []
        data = getStopWords.__init__()
        pathGeneral = self.path+'\\*.txta'
        files = glob.glob(pathGeneral)
        
        for name in files:
                with open(name,"r") as f:
                    txt = []
                    withoutStopWords = ""
                    txt = f.read().split()
                    f.close()
                    for i in range(1,len(txt)):
                        try: 
                            tmp = data.index(txt[i].lower())
                        except ValueError :
                                withoutStopWords  = withoutStopWords + txt[i] + "\n"
                                features.append(kok.stem(txt[i].lower())) 
                    change = open(name,"w")
                    change.write(name +" "+ withoutStopWords)
                    change.close()
    