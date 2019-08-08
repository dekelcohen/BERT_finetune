import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from sklearn import metrics

import catboost as cb


def get_dataset(path_to_csv):
    df = pd.read_csv(path_to_csv)
    return df

df = get_dataset('data/train.csv')


train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df['text'], df['label'])

tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), max_features=10000)
tfidf_vect_ngram.fit(df['text'])
xtrain_tfidf_ngram =  np.array(tfidf_vect_ngram.transform(train_x).todense(), dtype=np.float16) 
xvalid_tfidf_ngram =  np.array(tfidf_vect_ngram.transform(valid_x).todense(), dtype=np.float16) 

final_xtrain = xtrain_tfidf_ngram
final_xvalid = xvalid_tfidf_ngram
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)


# Extereme Gradient Boosting on Word Level TF IDF Vectors
accuracy = train_model(cb.CatBoostClassifier(custom_metric='Accuracy'), final_xtrain, train_y, final_xvalid)
print ("Catboost, NGram TF-IDF: %s" % (accuracy))

