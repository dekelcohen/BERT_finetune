import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from convert_data_util import map_labels,params 

    

#def convert_csv(params):
    
df = pd.read_csv(params.PATH_TO_TRAIN_CSV, sep=params.CSV_SEP) # Mac: may need , lineterminator='\r')
df = df.fillna('')
df = map_labels(df,params.LABEL_FIELD_NAME,params.LABELS_MAP)

le = LabelEncoder() 
# Creating train and dev dataframes according to BERT
df_bert = pd.DataFrame({'user_id':df[params.ROWID_FIELD_NAME],
            'label':le.fit_transform(df[params.LABEL_FIELD_NAME]),
            'alpha':['a']*df.shape[0],
            'text':df[params.TEXT_FIELDS].apply(lambda colVal: ' '.join(colVal), axis=1)})
 

df_bert_train = df_bert 

# If test set path argument is given, read and convert to BERT, if TEST_SIZE = 0.1 or 0.2 is given, split a test set
if params.PATH_TO_TEST_CSV is not None:
    # Creating test dataframe according to BERT
    df_test = pd.read_csv(params.PATH_TO_TEST_CSV, sep=params.CSV_SEP)
    df_bert_test = pd.DataFrame({'User_ID':df_test[params.ROWID_FIELD_NAME],
                 'text':df_test[params.TEXT_FIELDS].apply(lambda colVal: ' '.join(colVal), axis=1)})
elif params.TEST_SIZE > 0:
    df_bert_train, df_bert_test = train_test_split(df_bert, test_size=params.TEST_SIZE)
else:
    raise Exception('Error: Missing required parameter: Must specify either PATH_TO_TEST_CSV or TEST_SIZE (0.1)')

# Create train, dev split
df_bert_train, df_bert_dev = train_test_split(df_bert_train, test_size=params.DEV_SIZE)    
 
# Saving dataframes to .tsv format as required by BERT
if not os.path.exists('data'):
    os.makedirs('data')
    
df_bert_train.to_csv('data/train.tsv', sep='\t', index=False, header=False)
df_bert_dev.to_csv('data/dev.tsv', sep='\t', index=False, header=False)
df_bert_test.to_csv('data/expected_test.tsv', sep='\t', index=False, header=True)
df_bert_test = df_bert_test.drop(['label', 'alpha'], axis=1)
df_bert_test.to_csv('data/test.tsv', sep='\t', index=False, header=True)