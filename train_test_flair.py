# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 18:30:50 2019

@author: Dekel
"""
from pathlib import Path
from flair.data_fetcher import NLPTaskDataFetcher
from flair.data import Corpus
# from flair.datasets import TREC_6
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

# Set folder to output model and training metrics 
results_folder = 'enorn_6_folders_inbox'
# 1. get the corpus (either from flair datasets or a custom)
# corpus: Corpus = TREC_6()

# TODO: Change to ClassificationCorpus - see https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_6_CORPUS.md
corpus: Corpus = NLPTaskDataFetcher.load_classification_corpus(Path('./flair_data'), test_file='test.tsv', dev_file='dev.tsv', train_file='train.tsv')

# 2. create the label dictionary
label_dict = corpus.make_label_dictionary()

# 3. make a list of word embeddings
word_embeddings = [WordEmbeddings('glove'),

                   # comment in flair embeddings for state-of-the-art results
                   # FlairEmbeddings('news-forward'),
                   # FlairEmbeddings('news-backward'),
                   ]

# 4. initialize document embedding by passing list of word embeddings
# Can choose between many RNN types (GRU by default, to change use rnn_type parameter)
document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(word_embeddings,
                                                                     hidden_size=512,
                                                                     reproject_words=True,
                                                                     reproject_words_dimension=256,
                                                                     )

# 5. create the text classifier
classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

# 6. initialize the text classifier trainer
trainer = ModelTrainer(classifier, corpus)

# 7. start the training
trainer.train('resources/taggers/'+results_folder,
              learning_rate=0.1,
              mini_batch_size=32,
              anneal_factor=0.5,
              patience=5,
              max_epochs=150)

# 8. plot training curves (optional)
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves('resources/taggers/'+results_folder+'/loss.tsv')
plotter.plot_weights('resources/taggers/'+results_folder+'/weights.txt')

#### This is how metrics output looks like
#2019-06-04 11:37:58,523 ----------------------------------------------------------------------------------------------------
#2019-06-04 11:38:01,232 ----------------------------------------------------------------------------------------------------
#2019-06-04 11:38:01,232 Testing using best model ...
#2019-06-04 11:38:01,233 loading file resources/taggers/ag_news/best-model.pt
#2019-06-04 11:38:02,813 0.916   0.916   0.916
#2019-06-04 11:38:02,813
#MICRO_AVG: acc 0.845 - f1-score 0.916
#MACRO_AVG: acc 0.8359 - f1-score 0.9099666666666666
#ABBR       tp: 8 - fp: 1 - fn: 1 - tn: 490 - precision: 0.8889 - recall: 0.8889 - accuracy: 0.8000 - f1-score: 0.8889
#DESC       tp: 135 - fp: 19 - fn: 3 - tn: 343 - precision: 0.8766 - recall: 0.9783 - accuracy: 0.8599 - f1-score: 0.9247
#ENTY       tp: 75 - fp: 5 - fn: 19 - tn: 401 - precision: 0.9375 - recall: 0.7979 - accuracy: 0.7576 - f1-score: 0.8621
#HUM        tp: 59 - fp: 4 - fn: 6 - tn: 431 - precision: 0.9365 - recall: 0.9077 - accuracy: 0.8551 - f1-score: 0.9219
#LOC        tp: 77 - fp: 10 - fn: 4 - tn: 409 - precision: 0.8851 - recall: 0.9506 - accuracy: 0.8462 - f1-score: 0.9167
#NUM        tp: 104 - fp: 3 - fn: 9 - tn: 384 - precision: 0.9720 - recall: 0.9204 - accuracy: 0.8966 - f1-score: 0.9455
