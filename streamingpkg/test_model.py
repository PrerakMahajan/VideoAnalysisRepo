
import pandas as pd
#import nltk
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.stem import WordNetLemmatizer,SnowballStemmer
from nltk.stem.porter import *
nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle



'''
RETRIEVE_DATA takes a CSV file as the input and returns the corresponding arrays of labels and data as output. 
Name - Name of the csv file 
Train - If train is True, both the data and labels are returned. Else only the data naivebayess returned 
'''

stemmer = PorterStemmer()


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    clean_words = [lemmatize_stemming(token) for token in gensim.utils.simple_preprocess(text) if
                   token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3]
    return ' '.join(clean_words)


def retrieve_data(name="C:\\Users\\sourabh.vermaLA\\PycharmProjects\\VideoAnalysisRepo\\streamingpkg\\train.csv", train=True):
    data = pd.read_csv(name, header=0, encoding='ISO-8859-1')
    X = data['text']
    Y = data['polarity']

    if train:
        Y = data['polarity']
        return X, Y

    return X,Y

def oversample(x_feature):
    sm = SMOTE(random_state=12, ratio=1.0)
    x_res, y_res = sm.fit_sample(x_feature, Ytrain)
    print(pd.DataFrame(x_res.todense(),columns=tvect.get_feature_names()))
    #print("OVERSAMPLE :::",Ytrain.value_counts(), np.bincount(y_res))
    return x_res,y_res

'''
def remove_stopwords(sentence, stopwords,r):
    corpus=[]
    for i in range(0,r):
        sentencewords = sentence[i].split()
        resultwords = [word for word in sentencewords if word.lower() not in set(stopwords.words('english'))]
        result = ' '.join(resultwords)
        corpus.append(result)
    return corpus
'''


df = pd.read_csv('train10Videos.csv', header=0, encoding='ISO-8859-1')
#print("Value counts are mentioned below")
#print(df['polarity'].value_counts())

if __name__ == "__main__":
    import time
    start = time.time()

    [Xtrain_text, Ytrain] = retrieve_data(name="train10Videos.csv", train=True)
    Xtrain_preprcs_text = Xtrain_text.map(preprocess)

    Xtest_text,y_test = retrieve_data(name="test.csv", train=False)


'''
    #print("Test Data \n",Xtest_text)
    Xtest_text1 = Xtest_text

    # print("COUNT**********",c)
    # Xtest_text = remove_stopwords(Xtest_text, stopwords, c)
    #print("STOP WORDS REMOVED FROM TEST DATA::::: ", Xtest_text)
    
    r=Xtrain_text.shape[0]
    #print("COUNT**********",r)
    Xtrain_text=remove_stopwords(Xtrain_text,stopwords,r)
    print("STOP WORDS REMOVED from TRAIN DATA::::: ",Xtrain_text)
    
'''

#print("\nRetrieved the training data. Now will retrieve the test data in the required format")


#vectorizer = CountVectorizer(stop_words='english')

#######

vect = CountVectorizer(ngram_range=(1,2))
train_features = vect.fit_transform(Xtrain_preprcs_text)
#print(pd.DataFrame(train_features.toarray(),columns=vect.get_feature_names()))
test_features = vect.transform(Xtest_text)

######
tvect = TfidfVectorizer(ngram_range=(1,2))
train_features1 = tvect.fit_transform(Xtrain_preprcs_text)
#x_res,y_res=oversample(train_features1)
#print(pd.DataFrame(train_features.toarray(),columns=vect.get_feature_names()))
test_features1 = tvect.transform(Xtest_text)




#import pickle
nb = MultinomialNB()
rn=RandomForestClassifier(n_estimators=100,criterion='gini',n_jobs=-1)
x_res,y_res=oversample(train_features)
x_res1,y_res1=oversample(train_features1)
######
nbmodel = nb.fit(x_res,y_res)



######
#nbmodel1 = nb.fit(train_features1, Ytrain)
nbmodel1 = nb.fit(x_res1,y_res1)


with open('classifier.pickle','wb') as f:
    pickle.dump(nbmodel,f)

with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(tvect,f)

#loaded_model = pickle.load(open("fn.txt", 'rb'))
# Now we can use the model to predict classifications for our test features.

#####
predictions = nbmodel.predict_proba(test_features)
print("PREDICTIONS:::::",predictions)
#print(confusion_matrix(y_test,predictions))

#####
predictions1 = nbmodel1.predict_proba(test_features1)
#print(confusion_matrix(y_test,predictions1))
#predictions = loaded_model.predict(test_features)

# Convert the prediction to panda series


predSeries = pd.Series(predictions1)
chaResult = predSeries.replace(0, "It's a Challenge").replace(1, "It's not a challenge")

print("----------------Model Result of Text Analytics-----------------------------")
print("\n")
print("Text :" + Xtest_text)
print("\n")
print("Prediction :" + chaResult)
print("\n")

print("----------------SENTIMENT ANALYSIS STARTED ON GIVEN TEXT--------------------")
print("\n")