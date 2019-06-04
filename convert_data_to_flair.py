import os
import pandas as pd
from sklearn.model_selection import train_test_split
from convert_data_util import map_labels,get_default_params,write_dfs_to_csv

    
params = get_default_params()
params.PATH_TO_OUTPUT_DATA = 'flair_data'

def get_flair_df_from_csv(path_to_csv,params):
    df = pd.read_csv(path_to_csv, sep=params.CSV_SEP) # Mac: may need , lineterminator='\r')
    df = df.fillna('')
    df = map_labels(df,params.LABEL_FIELD_NAME,params.LABELS_MAP)
    df_flair = pd.DataFrame({'label': '__label__' + df[params.LABEL_FIELD_NAME],
            'text':df[params.TEXT_FIELDS].apply(lambda colVal: ' '.join(colVal), axis=1)})
    return df_flair

df = get_flair_df_from_csv(params.PATH_TO_TRAIN_CSV,params)
    
# Creating train and dev dataframes according to Flair
# TODO: label_ transform 
 

df_train = df

# If test set path argument is given, read and convert to BERT, if TEST_SIZE = 0.1 or 0.2 is given, split a test set
if params.PATH_TO_TEST_CSV is not None:
    # Creating test dataframe according to BERT
    df_test = get_flair_df_from_csv(params.PATH_TO_TEST_CSV,params)
elif params.TEST_SIZE > 0:
    df_train, df_test = train_test_split(df_train, test_size=params.TEST_SIZE)
else:
    raise Exception('Error: Missing required parameter: Must specify either PATH_TO_TEST_CSV or TEST_SIZE (0.1)')

# Create train, dev split
df_train, df_dev = train_test_split(df_train, test_size=params.DEV_SIZE)    

params.df_train = df_train
params.df_dev = df_dev
params.df_test = df_test

write_dfs_to_csv(params)    