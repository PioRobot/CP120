import nltk
nltk.download('punkt')
import pickle
import json
import numpy as np
import random

ignore_words = ['?', '!',',','.', "'s", "'m"]
import tensorflow
from data_preprocessing import get_stem_words
model=tensorflow.keras.models.load_model("./chatbot_model.h5")
wordsFriday13=json.loads(open("./intents.json").read())
wordRoot=pickle.load(open("./words.pkl","rb"))
taglist=pickle.load(open("./clases.pkl","rb"))

def preProcessUserImput(user_imput):
    wordT=nltk.word_tokenize(user_imput)
    rootW=get_stem_words(wordT,ignore_words)
    aOrg=sorted(list(set(rootW)))
    bagW=[]
    bagWords=[]
    for i in wordRoot:
        if i in aOrg:
            bagWords.append(1)
        else:
            bagWords.append(0)
    
    bagW.append(bagWords)
    return np.array(bagW)

def prediction_bot(user_imput):
    iMp=preProcessUserImput(user_imput)
    predVar=model.predict(iMp)
    valA=np.argmax(predVar[0])
    return valA

def botReply(user_imput):
    predBot=prediction_bot(user_imput)
    predClass=taglist[predBot]
    for j in wordsFriday13["intents"]:
        if j["tag"]==predClass:
            botReplY=random.choice(j["responses"])
            return botReplY

print("hello,i am not bayMax and i am not your personal medical assistent")
while True:
    user_imput=input("write your message now!  🔫 :")
    print("user input:",user_imput)
    reply=botReply(user_imput)
    print("answer 🔫 :",reply)
            
        
    
            