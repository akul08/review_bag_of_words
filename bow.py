import pandas as pd
import re
from bs4 import BeautifulSoup as bs
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def review_to_words(raw_review):
    review_text = bs(raw_review, "html.parser").get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words('english'))
    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)

train = pd.read_csv(
    'labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)

clean_train_reviews = [
    review_to_words(train['review'][i]) for i in range(train['review'].size)]

vectorizer = CountVectorizer(
    analyzer='word', tokenizer=None, preprocessor=None,
    stop_words=None, max_features=1500)

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()
# print vocab

dist = np.sum(train_data_features, axis=0)

# to print count of each vocab
# for tag, count in zip(vocab, dist):
#     print count, tag

forest = RandomForestClassifier(n_estimators=300)

forest = forest.fit(train_data_features, train['sentiment'])

test = pd.read_csv('testData.tsv', header=0, delimiter='\t', quoting=3)

clean_test_review = [
    review_to_words(test['review'][i]) for i in range(len(test['review']))]

test_data_features = vectorizer.transform(clean_test_review)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)

output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)
