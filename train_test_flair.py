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
#2019-06-04 13:10:05,359 Testing using best model ...
#2019-06-04 13:10:05,359 loading file resources/taggers/enorn_6_folders_inbox/best-model.pt
#2019-06-04 13:10:12,561 0.7809  0.7809  0.7809
#2019-06-04 13:10:12,561
#MICRO_AVG: acc 0.6406 - f1-score 0.7809
#MACRO_AVG: acc 0.6321 - f1-score 0.7731
#DontSave   tp: 53 - fp: 13 - fn: 26 - tn: 86 - precision: 0.8030 - recall: 0.6709 - accuracy: 0.5761 - f1-score: 0.7310
#Save       tp: 86 - fp: 26 - fn: 13 - tn: 53 - precision: 0.7679 - recall: 0.8687 - accuracy: 0.6880 - f1-score: 0.8152
#2019-06-04 13:10:12,561 ----------------------------------------------------------------------------------------------------
