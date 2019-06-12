import numpy as np
import pandas as pd
from sklearn import model_selection
from autokeras import TextClassifier

def convert_labels_to_one_hot(labels, num_labels):
    one_hot = np.zeros((len(labels), num_labels))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot


def get_dataset(path_to_csv):
    df = pd.read_csv(path_to_csv)
    return df

df = get_dataset('data/train.csv')


x_train, x_test, y_train, y_test = model_selection.train_test_split(df['text'], df['label'])

y_train = convert_labels_to_one_hot(y_train.values, num_labels=2)
y_test = convert_labels_to_one_hot(y_test.values, num_labels=2)

def train_model(clf, x_train, y_train, x_test, y_test):
    # print(len(label[-1]))
    # fit the training dataset on the classifier
    clf.fit(x=x_train, y=y_train, time_limit=1 * 60 * 60) # time_limit in seconds
    
    # classifier.final_fit(x=feature_vector_train, y=label)    
    # predict the labels on validation dataset
    print("Classification accuracy is : ", 100 * clf.evaluate(x_test, y_test), "%")
    
    


# , periodic_checkpoint_folder='checkpoint'
accuracy = train_model(TextClassifier(verbose = True), x_train.values, y_train, x_test.values, y_test)
print ("TPOT, NGram TF-IDF: %s" % (accuracy))
