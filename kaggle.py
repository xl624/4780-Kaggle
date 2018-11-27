import preprocessor as p
import pandas as pd
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import nltk
import datetime
from vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import normalize
import re
from nltk.tag import StanfordPOSTagger, StanfordNERTagger
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.ensemble import GradientBoostingClassifier

### need to set java_path, otherwise ner tagging & pos tagging would fail
java_path = "C:/Program Files/Java/jdk1.8.0_171/bin/java.exe"
os.environ['JAVAHOME'] = java_path
jar = "stanford-ner-2018-10-16\\stanford-ner.jar"

### some simple test to determine what features to include
stop_words = set(stopwords.words('english'))
p.set_options(p.OPT.URL, p.OPT.EMOJI,p.OPT.MENTION,p.OPT.HASHTAG,p.OPT.RESERVED,p.OPT.SMILEY,p.OPT.NUMBER)
data = pd.read_csv("train.csv")
data['tokenized'] = data['text'].apply(p.tokenize)
print("Average num char when y = 1: ", np.mean(data.loc[data['label']==1,'text'].apply(len)))
print("Average num char when y = -1: ", np.mean(data.loc[data['label']==-1,'text'].apply(len)))
print("Average num words when y = 1: ", np.mean(data.loc[data['label']==1,'text'].apply(lambda x: len(x.split()))))
print("Average num words when y = -1: ", np.mean(data.loc[data['label']==-1,'text'].apply(lambda x: len(x.split()))))
print("Average word length when y = 1: ", \
      np.mean(data.loc[data['label']==1,'text'].apply(lambda x: np.mean([len(a) for a in x.split()]))))
print("Average word length when y = -1: ", \
      np.mean(data.loc[data['label']==-1,'text'].apply(lambda x: np.mean([len(a) for a in x.split()]))))
print("Average num uppercase word when y = 1: ", \
      np.mean(data.loc[data['label']==1,'text'].apply(lambda x: len(re.findall('\s([A-Z][A-Z]+)', x)))))
print("Average num uppercase word when y = -1: ", \
      np.mean(data.loc[data['label']==-1,'text'].apply(lambda x: len(re.findall('\s([A-Z][A-Z]+)', x)))))
print("Percent whole sentence upper when y = 1: ", \
      np.mean(data.loc[data['label']==1,'tokenized'].apply(lambda x: x.isupper())))
print("Percent whole sentence upper when y = -1: ", \
      np.mean(data.loc[data['label']==-1,'tokenized'].apply(lambda x: x.isupper())))
print("Average num url when y = 1: ", np.mean(data.loc[data['label']==1,'tokenized'].apply(lambda x: x.count('$URL$'))))
print("Average num url when y = -1: ", np.mean(data.loc[data['label']==-1,'tokenized'].apply(lambda x: x.count('$URL$'))))
print("Average num mention when y = 1: ", \
      np.mean(data.loc[data['label']==1,'tokenized'].apply(lambda x: x.count('$MENTION$'))))
print("Average num mention when y = -1: ", \
      np.mean(data.loc[data['label']==-1,'tokenized'].apply(lambda x: x.count('$MENTION$'))))
print("Average num hashtag when y = 1: ", \
      np.mean(data.loc[data['label']==1,'tokenized'].apply(lambda x: x.count('$HASHTAG$'))))
print("Average num hashtag when y = -1: ", \
      np.mean(data.loc[data['label']==-1,'tokenized'].apply(lambda x: x.count('$HASHTAG$'))))
print("Average num number when y = 1: ", \
      np.mean(data.loc[data['label']==1,'tokenized'].apply(lambda x: x.count('$NUMBER$'))))
print("Average num number when y = -1: ", \
      np.mean(data.loc[data['label']==-1,'tokenized'].apply(lambda x: x.count('$NUMBER$'))))
print("Average num smiley when y = 1: ", \
      np.mean(data.loc[data['label']==1,'tokenized'].apply(lambda x: x.count('$SMILEY$'))))
print("Average num smiley when y = -1: ", \
      np.mean(data.loc[data['label']==-1,'tokenized'].apply(lambda x: x.count('$SMILEY$'))))
print("Average num start with \" when y = 1: ", \
      np.mean(data.loc[data['label']==1,'tokenized'].apply(lambda x: len(re.findall('^"', x)))))
print("Average num start with \" when y = -1: ", \
      np.mean(data.loc[data['label']==-1,'tokenized'].apply(lambda x: len(re.findall('^"', x)))))
print("Average num start with @ when y = 1: ", \
      np.mean(data.loc[data['label']==1,'text'].apply(lambda x: len(re.findall('^@', x)))))
print("Average num start with @ when y = -1: ", \
      np.mean(data.loc[data['label']==-1,'text'].apply(lambda x: len(re.findall('^@', x)))))
print("Average num start with # when y = 1: ", \
      np.mean(data.loc[data['label']==1,'text'].apply(lambda x: len(re.findall('^#', x)))))
print("Average num start with # when y = -1: ", \
      np.mean(data.loc[data['label']==-1,'text'].apply(lambda x: len(re.findall('^#', x)))))
print("Average num ; when y = 1: ", np.mean(data.loc[data['label']==1,'text'].apply(lambda x: x.count(";"))))
print("Average num ; when y = -1: ", np.mean(data.loc[data['label']==-1,'text'].apply(lambda x: x.count(";"))))
print("Average num ! when y = 1: ", np.mean(data.loc[data['label']==1,'text'].apply(lambda x: x.count("!"))))
print("Average num ! when y = -1: ", np.mean(data.loc[data['label']==-1,'text'].apply(lambda x: x.count("!"))))
print("Average num \" when y = 1: ", np.mean(data.loc[data['label']==1,'text'].apply(lambda x: x.count("\""))))
print("Average num \" when y = -1: ", np.mean(data.loc[data['label']==-1,'text'].apply(lambda x: x.count("\""))))
print("Average num ... when y = 1: ", np.mean(data.loc[data['label']==1,'text'].apply(lambda x: x.count("..."))))
print("Average num ... when y = -1: ", np.mean(data.loc[data['label']==-1,'text'].apply(lambda x: x.count("..."))))
print("Average num . when y = 1: ", np.mean(data.loc[data['label']==1,'text'].apply(lambda x: x.count("."))))
print("Average num . when y = -1: ", np.mean(data.loc[data['label']==-1,'text'].apply(lambda x: x.count("."))))
print("Average num ? when y = 1: ", np.mean(data.loc[data['label']==1,'text'].apply(lambda x: x.count("?"))))
print("Average num ? when y = -1: ", np.mean(data.loc[data['label']==-1,'text'].apply(lambda x: x.count("?"))))
print("Average favorite when y = 1: ", np.mean(data.loc[data['label']==1,'favoriteCount']))
print("Average favorite when y = -1: ", np.mean(data.loc[data['label']==-1,'favoriteCount']))
print("Average retweet when y = 1: ", np.mean(data.loc[data['label']==1,'retweetCount']))
print("Average retweet when y = -1: ", np.mean(data.loc[data['label']==-1,'retweetCount']))
print("Average id when y = 1: ", np.mean(data.loc[data['label']==1,'id']))
print("Average id when y = -1: ", np.mean(data.loc[data['label']==-1,'id']))
print("Average stopword when y = 1: ", \
      np.mean(data.loc[data['label']==1,'text'].apply(lambda x: sum([a in stop_words for a in x.split()]))))
print("Average stopword when y = -1: ", \
      np.mean(data.loc[data['label']==-1,'text'].apply(lambda x: sum([a in stop_words for a in x.split()]))))
      

### read dataset      
data = pd.read_csv("train.csv")
data1 = pd.read_csv("test.csv")
data1['statusSource'] = 0
data = pd.concat([data,data1],axis=0)
data.index = np.arange(0,len(data))

### some feature extraction steps
def preprocess(data):
    data['created']=pd.to_datetime(data['created'], errors='coerce')
    data['dayInWeek'] = data['created'].dt.weekday
    data['hourInDay'] = data['created'].dt.hour
    data['tokenized'] = data['text'].apply(p.tokenize)
    data['numchar'] = data['text'].apply(len)
    data['numWords'] = data['text'].apply(lambda x: len(x.split()))
    data['avgWordLen'] =data['text'].apply(lambda x: np.mean([len(a) for a in x.split()]))
    data['wholeUpper'] = data['tokenized'].apply(lambda x: np.sum(x.isupper()))
    data['numURL'] = data['tokenized'].apply(lambda x: x.count('$URL$'))
    data['numHash'] = data['tokenized'].apply(lambda x: x.count('$HASHTAG$'))
    data['mention'] = data['tokenized'].apply(lambda x: x.count('$MENTION$'))
    data['startQuote']=data['tokenized'].apply(lambda x: len(re.findall('^"', x)))
    data['startMention']=data['tokenized'].apply(lambda x: len(re.findall('^@', x)))
    data['startHash']=data['tokenized'].apply(lambda x: len(re.findall('^#', x)))
    data['numComa']=data['text'].apply(lambda x: x.count(";"))
    data['numQuoSym']=data['text'].apply(lambda x: x.count("\""))
    data['numSmiley'] = data['tokenized'].apply(lambda x: x.count('$SMILEY$'))
    data['numStopWord']=data['text'].apply(lambda x: sum([a in stop_words for a in x.split()]))
    data['wordLength']=data['text'].apply(lambda x: np.mean([len(a) for a in x.split()]))
    data['posTagged']=data['tokenized'].apply(posTagging) # pos tagging, take very long to run
    data['nerTagged']=data['tokenized'].apply(nerTagging) # ner tagging, take very long to run
    return data

### convert name, place, organizations etc into some tag    
def nerTagging(text):
    jar = "stanford-ner-2018-10-16\\stanford-ner.jar"
    cla = "stanford-ner-2018-10-16\\classifiers\\english.all.3class.distsim.crf.ser.gz"
    st = StanfordNERTagger(cla,jar)
    ner = st.tag(text.split())
    after = ""
    for item in ner:
        if item[1] == 'O':
            after += item[0] + ' '
        else:
            after += item[1] + ' '
    return after

### tag the text with verb, noun blabla
def posTagging(text):
    jar = 'stanford-postagger-2018-10-16\\stanford-postagger.jar'
    model = "stanford-postagger-2018-10-16\\models\\english-bidirectional-distsim.tagger"
    st = StanfordPOSTagger(model,jar)
    pos = st.tag(text.split())
    after = ""
    for item in pos:
        after += item[1] + " "
    return after

### sentiment analysis which I didn't include in the final code
# data = apply_vader(data, 'text')
# data['created']=pd.to_datetime(data['created'], errors='coerce')
# def apply_vader(df, column):
    # sentiment = pd.DataFrame(df[column].apply(get_vader_scores))
    # unpacked = pd.DataFrame([d for idx, d in sentiment[column].iteritems()],
                            # index=sentiment.index)
    # unpacked['compound'] += 1
    # columns = {'neu': 'v_neutral', 'pos': 'v_positive', 'neg': 'v_negative'}
    # unpacked.rename(columns=columns, inplace=True)
    # return pd.concat([df, unpacked], axis=1)

data = preprocess(data)
# save the data as preprocess takes a long time to run, 
data = pd.to_csv("data.csv")


#data=pd.read_csv("data.csv")

# TfidfVectorizer generates features from text
# generate feature from orignal text
text = TfidfVectorizer(ngram_range=(1, 2))
a = text.fit_transform(data['text'])
colnames = text.get_feature_names()
a.todense().shape
idx = data.index
text = pd.DataFrame(a.todense(),columns=[colnames],index=idx)

# generate feature from orignal pos tags
pos = TfidfVectorizer(ngram_range=(1, 3))
a = pos.fit_transform(data['posTagged'])
colnames = pos.get_feature_names()
a.todense().shape
idx = data.index
pos = pd.DataFrame(a.todense(),columns=[colnames],index=idx)

# # generate feature from ner tags
ner = TfidfVectorizer(ngram_range=(1, 2))
a = ner.fit_transform(data['nerTagged'])
colnames = ner.get_feature_names()
a.todense().shape
idx = data.index
pos = pd.DataFrame(a.todense(),columns=[colnames],index=idx)

# concat features into one data frame
data = pd.concat([data,text,pos],axis=1)

# drop unwanted columns
X = data.drop(["nerTagged"],axis=1)
X = X.drop(['label'], axis=1)
X=X.drop(['favorited', 'replyToSID', 'truncated','screenName','isRetweet','retweeted'],axis=1)
X=X.drop(['id', 'text', 'favoriteCount','replyToSN','created','id.1','replyToUID',\
               "longitude",'latitude','tokenized','posTagged'],axis=1)
X = X.drop(['statusSource'],axis=1)
Y = pd.DataFrame(np.where(data['label'] == 1, 1,0))
X = np.squeeze(X.values)
X = normalize(X)
Y = Y.values.flatten()
cut = 1089
X_train = X[:cut,]
Y_train = Y[:cut]
gb = GradientBoostingClassifier(n_estimators=200,
                                        learning_rate=.1,
                                        max_depth=6,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        subsample=1,
                                        max_features=None
                                        ).fit(X_train, Y_train)
X_valid = X[cut:,]
predicted = gb.predict(X_valid)

predicted = pd.DataFrame(predicted)
predicted.to_csv("predicted.csv")
