import pickle
import pandas as pd

with open('classifier.pickle','rb') as f:
    clf=pickle.load(f)

with open('tfidfmodel.pickle', 'rb') as f:
    tfidf = pickle.load(f)

sample=["Technology is the biggest issue being faced by fintech today"]
#sample=["Technology is the biggest asset for a fintech"]
sample=tfidf.transform(sample).toarray()
res=pd.Series(clf.predict(sample))


print(res.replace(0, "It's a Challenge").replace(1, "It's not a challenge"))