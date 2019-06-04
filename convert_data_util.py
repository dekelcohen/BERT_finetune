import os
import copy
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
 
class MyObj():
    def __init__(self):
        pass

def_params = MyObj()

def_params.PATH_TO_TRAIN_CSV= 'D:/Dekel/Data/Text_py/emailinsight/pyScripts/data/enron_6_email_folders_Inboxes_KAMINSKI.tsv'
def_params.CSV_SEP='\t'
def_params.LABEL_FIELD_NAME = 'folderName'
def_params.ROWID_FIELD_NAME = 'updateId'
# The following fields are concat to a single text field for BERT input
def_params.TEXT_FIELDS = ['subject', 'body', 'from', 'fromDomain', 'to', 'cc' ]
def_params.PATH_TO_TEST_CSV = None
def_params.TEST_SIZE = 0.1
def_params.DEV_SIZE  = 0.1
def_params.LABELS_MAP = { 'Inbox' : 'DontSave','Notes inbox' : 'DontSave', 'default_mapping' : 'Save' } # manual mapping with default mapping
def_params.PATH_TO_OUTPUT_DATA = 'data'

def get_default_params():
    return copy.deepcopy(def_params)

def map_labels(df,label_col_name,labels_map):
    
    '''
    Map labels according to user supplied labels_map
    '''
    # TODO: Clone labels_map, removing default_mapping 
    dict_no_default = copy.deepcopy(labels_map)
    if 'default_mapping' in labels_map:
        del dict_no_default['default_mapping']
     
    dfCol = df[label_col_name].map(dict_no_default)
    if 'default_mapping' in labels_map:
        default_value = labels_map['default_mapping']
        dfCol = dfCol.fillna(default_value).astype(str)    
    df[label_col_name] = dfCol
    return df
    

def write_dfs_to_csv(params):
    out_path = params.PATH_TO_OUTPUT_DATA    
    # Saving dataframes to .tsv format in target format (label,text ...)
    if not os.path.exists(out_path):
        os.makedirs(out_path)        
    params.df_train.to_csv(os.path.join(out_path,'train.tsv'), sep='\t', index=False, header=False)    
    params.df_dev.to_csv(os.path.join(out_path,'dev.tsv'), sep='\t', index=False, header=False)
    params.df_test.to_csv(os.path.join(out_path,'test.tsv'), sep='\t', index=False, header=False)