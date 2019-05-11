# See Post  https://appliedmachinelearning.blog/2019/03/04/state-of-the-art-text-classification-using-bert-model-predict-the-happiness-hackerearth-challenge/
# clone BERT repo: git clone https://github.com/google-research/bert.git
# Use convert_data_to_bert.py to convert the input .CSV to BERT format and create train/dev/test split --> result in data/train.csv data/dev.csv data/test.csv in BERT format
# Download and unzip BERT model (ex: cased_L-12_H-768_A-12 - see command line params below) from https://github.com/google-research/bert#pre-trained-models
# wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
mkdir bert_output

python bert/run_classifier.py --task_name=cola --do_train=true --do_eval=true --do_predict=true --data_dir=./data/ --vocab_file=./cased_L-12_H-768_A-12/vocab.txt --bert_config_file=./cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=./cased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=400 --train_batch_size=8 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=./bert_output/ --do_lower_case=False