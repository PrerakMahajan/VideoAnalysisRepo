# AIM: TEXT MINING and SENTIMENT ANALYSIS of Youtube Videos
import stanfordnlp

# train_path = "imdb_tr.csv"
train_path = "train.csv"  # source data
test_path = "test/test.csv"  # test data for grade evaluation.

'''
Inpath - Path of the training samples 
Outpath - Path were the file has to be saved 
Name  - Name with which the file has to be saved 
Mix - Used for shuffling the data 
'''


def imdb_data_preprocess(inpath, outpath="./", name="train.csv", mix=False):
    import pandas as pd
    from pandas import DataFrame, read_csv
    import os
    import csv
    import numpy as np

    stopwords = open("stopwords.en.txt", 'r', encoding="ISO-8859-1").read()
    stopwords = stopwords.split("\n")

    indices = []
    text = []
    rating = []

    i = 0

    for filename in os.listdir(inpath + "pos"):
        data = open(inpath + "pos/" + filename, 'r', encoding="ISO-8859-1").read()
        data = remove_stopwords(data, stopwords)
        indices.append(i)
        text.append(data)
        rating.append("1")
        i = i + 1

    for filename in os.listdir(inpath + "neg"):
        data = open(inpath + "neg/" + filename, 'r', encoding="ISO-8859-1").read()
        data = remove_stopwords(data, stopwords)
        indices.append(i)
        text.append(data)
        rating.append("0")
        i = i + 1

    Dataset = list(zip(indices, text, rating))

    if mix:
        np.random.shuffle(Dataset)

    df = pd.DataFrame(data=Dataset, columns=['row_Number', 'text', 'polarity'])
    df.to_csv(outpath + name, index=False, header=True)

    pass


'''
REMOVE_STOPWORDS takes a sentence and the stopwords as inputs and returns the sentence without any stopwords 
Sentence - The input from which the stopwords have to be removed
Stopwords - A list of stopwords  
'''


def remove_stopwords(sentence, stopwords):
    sentencewords = sentence.split()
    resultwords = [word for word in sentencewords if word.lower() not in stopwords]
    result = ' '.join(resultwords)
    return result


'''
UNIGRAM_PROCESS takes the data to be fit as the input and returns a vectorizer of the unigram as output 
Data - The data for which the unigram model has to be fit 
'''


def unigram_process(data):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    vectorizer = vectorizer.fit(data)
    return vectorizer


'''
BIGRAM_PROCESS takes the data to be fit as the input and returns a vectorizer of the bigram as output 
Data - The data for which the bigram model has to be fit 
'''


def bigram_process(data):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    vectorizer = vectorizer.fit(data)
    return vectorizer


'''
TFIDF_PROCESS takes the data to be fit as the input and returns a vectorizer of the tfidf as output 
Data - The data for which the bigram model has to be fit 
'''


def tfidf_process(data):
    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer()
    transformer = transformer.fit(data)
    return transformer


'''
RETRIEVE_DATA takes a CSV file as the input and returns the corresponding arrays of labels and data as output. 
Name - Name of the csv file 
Train - If train is True, both the data and labels are returned. Else only the data inaivebayess returned 
'''


def retrieve_data(name="train.csv", train=True):
    import pandas as pd
    data = pd.read_csv(name, header=0, encoding='ISO-8859-1')
    X = data['text']

    if train:
        Y = data['polarity']
        return X, Y

    return X


'''
STOCHASTIC_DESCENT applies Stochastic on the training data and returns the predicted labels 
Xtrain - Training Data
Ytrain - Training Labels
Xtest - Test Data 
'''


def stochastic_descent(Xtrain, Ytrain, Xtest):
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(loss="hinge", penalty="l1", n_iter=20)
    print("SGD Fitting")
    clf.fit(Xtrain, Ytrain)
    print("SGD Predicting")
    Ytest = clf.predict(Xtest)
    return Ytest


'''
ACCURACY finds the accuracy in percentage given the training and test labels 
Ytrain - One set of labels 
Ytest - Other set of labels 
'''


def accuracy(Ytrain, Ytest):
    assert (len(Ytrain) == len(Ytest))
    num = sum([1 for i, word in enumerate(Ytrain) if Ytest[i] == word])
    n = len(Ytrain)
    print('###')
    print((num * 100) / n)
    return (num * 100) / n


'''
WRITE_TXT writes the given data to a text file 
Data - Data to be written to the text file 
Name - Name of the file 
'''


def write_txt(data, name):
    data = ''.join(str(word) for word in data)
    file = open(name, 'w')
    file.write(data)
    file.close()
    pass


if __name__ == "__main__":
    import time

    start = time.time()
    print("Preprocessing the training_data--")
    # imdb_data_preprocess(inpath=train_path, mix=True)
    print("Done with preprocessing. Now, will retreieve the training data in the required format")
    [Xtrain_text, Ytrain] = retrieve_data()
    #print([Xtrain_text, Ytrain])
    # import pandas as pd
    # [Row, Xtrain_text, Ytrain] = pd.read_csv("C:\\Users/shiva/Downloads/AI-Sentiment-Analysis-on-IMDB-Dataset-master/imdb_tr     #.csv" ,  names = ['Row', 'text', 'polarity'], header = None, encoding="ISO-8859-1")
    print("Retrieved the training data. Now will retrieve the test data in the required format")

    Xtest_text = retrieve_data(name=test_path, train=False)
    #print(Xtest_text)
    #print("Retrieved the test data. Now will initialize the model \n\n")


   #print ("-----------------------ANALYSIS ON THE INSAMPLE DATA (TRAINING DATA)---------------------------")

"""
   uni_vectorizer = unigram_process(Xtrain_text)
   print ("Fitting the unigram model")
   Xtrain_uni = uni_vectorizer.transform(Xtrain_text)
   print ("After fitting ")
   #print ("Applying the stochastic descent")
   #Y_uni = stochastic_descent(Xtrain_uni, Ytrain, Xtrain_uni)
   #print ("Done with  stochastic descent")
   #print ("Accuracy for the Unigram Model is ", accuracy(Ytrain, Y_uni))
   print ("\n")

   bi_vectorizer = bigram_process(Xtrain_text)
   print ("Fitting the bigram model")
   Xtrain_bi = bi_vectorizer.transform(Xtrain_text)
   print ("After fitting ")
   #print ("Applying the stochastic descent")
   #Y_bi = stochastic_descent(Xtrain_bi, Ytrain, Xtrain_bi)
   #print ("Done with  stochastic descent")
   #print ("Accuracy for the Bigram Model is ", accuracy(Ytrain, Y_bi))
   print ("\n")

   uni_tfidf_transformer = tfidf_process(Xtrain_uni)
   print ("Fitting the tfidf for unigram model")
   Xtrain_tf_uni = uni_tfidf_transformer.transform(Xtrain_uni)
   print ("After fitting TFIDF")
   #print ("Applying the stochastic descent")
   #Y_tf_uni = stochastic_descent(Xtrain_tf_uni, Ytrain, Xtrain_tf_uni)
   #print ("Done with  stochastic descent")
   #print ("Accuracy for the Unigram TFIDF Model is ", accuracy(Ytrain, Y_tf_uni))
   print ("\n")


   bi_tfidf_transformer = tfidf_process(Xtrain_bi)
   print ("Fitting the tfidf for bigram model")
   Xtrain_tf_bi = bi_tfidf_transformer.transform(Xtrain_bi)
   print ("After fitting TFIDF")
   #print ("Applying the stochastic descent")
   #Y_tf_bi = stochastic_descent(Xtrain_tf_bi, Ytrain, Xtrain_tf_bi)
   #print ("Done with  stochastic descent")
   #print ("Accuracy for the Unigram TFIDF Model is ", accuracy(Ytrain, Y_tf_bi))
   print ("\n")
   """

from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import multilayer_perceptron
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn import metrics

# Generate counts from text using a vectorizer.  There are other vectorizers available, and lots of options you can set.
# This performs our step of computing word counts.
# vectorizer = VectorizerMixin(stop_words='english')

vectorizer = CountVectorizer(stop_words='english')
train_features = vectorizer.fit_transform(Xtrain_text)
test_features = vectorizer.transform(Xtest_text)


print("---------------BUILDING NAIVE BAYES MODEL COMPLETED----------------------")
print("\n")
# Fit a naive bayes model to the training data.
# This will train the model using the word counts we computer, and the existing classifications in the training set.
nb = MultinomialNB()
nbmodel = nb.fit(train_features, Ytrain)
#pickle.dump(nbmodel, open("fn.txt", 'wb'))

loaded_model = pickle.load(open("fn.txt", 'rb'))
# Now we can use the model to predict classifications for our test features.
#predictions = nbmodel.predict(test_features)
predictions = loaded_model.predict(test_features)

# Convert the prediction to panda series
import pandas as pd

predSeries = pd.Series(predictions)
chaResult = predSeries.replace(0, "It's a Challenge").replace(1, "It's not a challenge")

print("----------------Model Result of Text Analytics-----------------------------")
print("\n")
print("Text :" + Xtest_text)
print("\n")
print("Prediction :" + chaResult)
print("\n")

print("----------------SENTIMENT ANALYSIS STARTED ON GIVEN TEXT--------------------")
print("\n")

# Paralleldots to take out Npe, and some more results
# import paralleldots
# paralleldots.set_api_key("1XLNThDtfEr52Sltn7zDJWShmqJU2NbfHmF5eSyu1Fo")

# f = open("C:\\Users/shiva/Downloads/AI-Sentiment-Analysis/unigram.output.txt",'w' , encoding="ISO-8859-1")

# Taking out sentiment
# sent = paralleldots.sentiment(Xtest_text)
# resultSent = pd.DataFrame.from_dict(sent)
# print("Sentiment of Given Text:")
# print(resultSent)
# print("\n")

# Taking out NER Information
# ner = paralleldots.ner(Xtest_text)
# resultNER = pd.DataFrame.from_dict(ner)
# print("NER Information of Given Text:")
# print(resultNER)

# Compute the error.  It is slightly different from our model because the internals of this process work differently from our implementation.

# fpr, tpr, thresholds = metrics.roc_curve(Ytrain, predictions, pos_label=1)

# print("Multinomial naive bayes AUC: {0}".format(metrics.auc(fpr, tpr)))
"""
   print ("-----------------------ANALYSIS ON THE TEST DATA ---------------------------")
   print ("Unigram Model on the Test Data--")
   Xtest_uni = uni_vectorizer.transform(Xtest_text)
   print ("Applying the stochastic descent")
   Ytest_uni = stochastic_descent(Xtrain_uni, Ytrain, Xtest_uni)
   write_txt( Ytest_uni, name="unigram.output.txt")
   print ("Done with  descent")
   print ("\n")


   print ("Bigram Model on the Test Data--")
   Xtest_bi = bi_vectorizer.transform(Xtest_text)
   print ("Applying the stochastic descent")
   Ytest_bi = stochastic_descent(Xtrain_bi, Ytrain, Xtest_bi)
   write_txt( Ytest_bi, name="bigram.output.txt")
   print ("Done with  stochastic descent")
   print ("\n")

   print ("Unigram TF Model on the Test Data--")
   Xtest_tf_uni = uni_tfidf_transformer.transform(Xtest_uni)
   print ("Applying the stochastic descent")
   Ytest_tf_uni = stochastic_descent(Xtrain_tf_uni, Ytrain, Xtest_tf_uni)
   write_txt( Ytest_tf_uni, name="unigramtfidf.output.txt")
   print ("Done with  stochastic descent")
   print ("\n")

   print ("Bigram TF Model on the Test Data--")
   Xtest_tf_bi = bi_tfidf_transformer.transform(Xtest_bi)
   print ("Applying the stochastic descent")
   Ytest_tf_bi = stochastic_descent(Xtrain_tf_bi, Ytrain, Xtest_tf_bi)
   write_txt( Ytest_tf_bi, name="bigramtfidf.output.txt")
   print ("Done with  stochastic descent")
   print ("\n")


   print ("Total time taken is ", time.time()-start, " seconds") 
pass """""