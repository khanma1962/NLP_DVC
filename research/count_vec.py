
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "apple ball cat",
    "ball cat dog elephant",
    "ball cat dog"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
# print(f"The value of {vectorizer.get_feature_names()}")
# print(f"Print x.toarray is {X.toarray()}")

max_feature = 300
ngrams = 2

vectorizer2 = CountVectorizer(max_features=  max_feature, ngram_range= (1, ngrams) )
X2 = vectorizer2.fit_transform(corpus)
print(f"The value of {vectorizer2.get_feature_names()}")
print(f"Print x.toarray is {X2.toarray()}")


