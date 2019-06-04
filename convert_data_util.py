import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
 
class MyObj():
    def __init__(self):
        pass

params = MyObj()

params.PATH_TO_TRAIN_CSV= 'D:/Dekel/Data/Text_py/emailinsight/pyScripts/data/enron_6_email_folders_Inboxes_KAMINSKI.tsv'
params.CSV_SEP='\t'
params.LABEL_FIELD_NAME = 'folderName'
params.ROWID_FIELD_NAME = 'updateId'
# The following fields are concat to a single text field for BERT input
params.TEXT_FIELDS = ['subject', 'body', 'from', 'fromDomain', 'to', 'cc' ]
params.PATH_TO_TEST_CSV = None
params.TEST_SIZE = 0.1
params.DEV_SIZE  = 0.1
params.LABELS_MAP = { 'Inbox' : 'DontSave','Notes inbox' : 'DontSave', 'default_mapping' : 'Save' } # manual mapping with default mapping


def map_labels(df,label_col_name,labels_map):
    import copy
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
    

