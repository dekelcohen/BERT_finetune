import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

 
PATH_TO_TRAIN_CSV= 'sample_input_data/enron_6_email_folders_KAMINSKI.tsv'
CSV_SEP='\t'
LABEL_FIELD_NAME = 'folderName'
ROWID_FIELD_NAME = 'updateId'
# The following fields are concat to a single text field for BERT input
TEXT_FIELDS = ['subject', 'body', 'from', 'fromDomain', 'to', 'cc' ]
PATH_TO_TEST_CSV = None
TEST_SIZE = 0.1
DEV_SIZE  = 0.1

 
df = pd.read_csv(PATH_TO_TRAIN_CSV, sep=CSV_SEP) # Mac: may need , lineterminator='\r')
df = df.fillna('')

le = LabelEncoder() 
# Creating train and dev dataframes according to BERT
df_bert = pd.DataFrame({'user_id':df[ROWID_FIELD_NAME],
            'label':le.fit_transform(df[LABEL_FIELD_NAME]),
            'alpha':['a']*df.shape[0],
            'text':df[TEXT_FIELDS].apply(lambda colVal: ' '.join(colVal), axis=1)})
 

df_bert_train = df_bert 

# If test set path argument is given, read and convert to BERT, if TEST_SIZE = 0.1 or 0.2 is given, split a test set
if PATH_TO_TEST_CSV is not None:
    # Creating test dataframe according to BERT
    df_test = pd.read_csv(PATH_TO_TEST_CSV, sep=CSV_SEP)
    df_bert_test = pd.DataFrame({'User_ID':df_test[ROWID_FIELD_NAME],
                 'text':df_test[TEXT_FIELDS].apply(lambda colVal: ' '.join(colVal), axis=1)})
elif TEST_SIZE > 0:
    df_bert_train, df_bert_test = train_test_split(df_bert, test_size=TEST_SIZE)
else:
    raise Exception('Error: Missing required parameter: Must specify either PATH_TO_TEST_CSV or TEST_SIZE (0.1)')

# Create train, dev split
df_bert_train, df_bert_dev = train_test_split(df_bert_train, test_size=0.1)    
 
# Saving dataframes to .tsv format as required by BERT
if not os.path.exists('data'):
    os.makedirs('data')
    
df_bert_train.to_csv('data/train.tsv', sep='\t', index=False, header=False)
df_bert_dev.to_csv('data/dev.tsv', sep='\t', index=False, header=False)
df_bert_test.to_csv('data/expected_test.tsv', sep='\t', index=False, header=True)
df_bert_test = df_bert_test.drop(['label', 'alpha'], axis=1)
df_bert_test.to_csv('data/test.tsv', sep='\t', index=False, header=True)