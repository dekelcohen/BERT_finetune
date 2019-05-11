# BERT_finetune
Fine tuning BERT for text and email classification 

# Getting Started 
## Clone this repo 
git clone https://github.com/dekelcohen/BERT_finetune.git
cd BERT_finetune
## Download and unzip BERT model (ex: cased_L-12_H-768_A-12 - see command line params below) from https://github.com/google-research/bert#pre-trained-models
wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
unzip cased_L-12_H-768_A-12.zip 

## clone BERT repo: 
git clone https://github.com/google-research/bert.git
## Use convert_data_to_bert.py to convert the input .CSV to BERT format and create train/dev/test split --> result in data/train.csv data/dev.csv data/test.csv in BERT format

## Edit paramters at the top of the file (path to csv ... or leave the defaults for sample enron data)
python convert_data_to_bert.py

## Edit bert/run_classifier.py ColaProcessor.get_labels (line 456) to reflect all labels created 0-<num of labels -1>. e.g for 6 labels: return ["0", "1", "2", "3","4","5"]
chmod +x train_classifier.sh
./train_classifier.sh 
# See Post  https://appliedmachinelearning.blog/2019/03/04/state-of-the-art-text-classification-using-bert-model-predict-the-happiness-hackerearth-challenge/