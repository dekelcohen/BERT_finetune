import os
import pandas as pd
from sklearn.model_selection import train_test_split
from convert_data_util import map_labels,get_default_params,split_train_dev_test,write_dfs_to_csv

    
params = get_default_params()
params.PATH_TO_OUTPUT_DATA = 'azureml_data'

def get_azureml_df_from_csv(path_to_csv,params):
    df = pd.read_csv(path_to_csv, sep=params.CSV_SEP) # Mac: may need , lineterminator='\r')
    df = df.fillna('')
    df = map_labels(df,params.LABEL_FIELD_NAME,params.LABELS_MAP)
    df_azureml = pd.DataFrame({'label': df[params.LABEL_FIELD_NAME],
            'text':df[params.TEXT_FIELDS].apply(lambda colVal: ' '.join(colVal).replace(',',' '), axis=1)})
    return df_azureml

df = get_azureml_df_from_csv(params.PATH_TO_TRAIN_CSV,params)
    

out_path = params.PATH_TO_OUTPUT_DATA    
# Saving dataframes to .tsv format in target format (label,text ...)
if not os.path.exists(out_path):
    os.makedirs(out_path)
    
df.to_csv(os.path.join(out_path,'train.csv'), sep=',', index=False, header=True)    