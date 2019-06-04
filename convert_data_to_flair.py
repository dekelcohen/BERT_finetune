import os
import pandas as pd
from sklearn.model_selection import train_test_split
from convert_data_util import map_labels,get_default_params,split_train_dev_test,write_dfs_to_csv

    
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
 

split_train_dev_test(df,params,get_flair_df_from_csv)
write_dfs_to_csv(params)    