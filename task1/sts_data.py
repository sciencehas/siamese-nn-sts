
```python
import pandas as pd
from preprocess import Preprocess
import logging
import torch
from dataset import STSDataset  # Now importing new STSDataset for Combined dataset
from datasets import load_dataset
import torchtext
from torch.utils.data import DataLoader
import spacy
import numpy as np
from torchtext.legacy import data
from gspread_dataframe import get_as_dataframe, set_with_dataframe  # for pandas dataframe conversion 
spacy.load('en_core_web_sm')

logging.basicConfig(level=logging.INFO)

"""
For loading STS data loading and preprocessing for the Combined dataset
"""

class STSData:
    def __init__(
        self,
        dataset_name,
        columns_mapping,
        stopwords_path="stopwords-en.txt",
        model_name="lstm",
        max_sequence_len=20,
        normalization_const=5.0,
        normalize_labels=False,
    ):
        """
        Loads data into memory and create vocabulary from text field.
        """
        self.normalization_const = normalization_const
        self.normalize_labels = normalize_labels
        self.model_name = model_name
        self.max_sequence_len = max_sequence_len
        self.dataset_name = dataset_name
        
        # Load Combined dataset from Google Sheets into dataframe
        self.load_data(dataset_name, columns_mapping, stopwords_path)
        self.columns_mapping = columns_mapping
        self.create_vocab()

    def load_data(self, dataset_name, columns_mapping, stpwds_file_path):
        """
        Reads Combined dataset from Google Sheets into dataframe, replacing the old function to load from file
        """
        logging.info("Loading and preprocessing data...")

        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            'client_secret.json', scope)
        client = gspread.authorize(creds)

        # Replace 'Sick' with the actual file name 'Combined_Dataset'
        sheet = client.open('Combined_Dataset').sheet1

        # Pandas dataframe conversion
        df = get_as_dataframe(sheet)

        # Splitting dataset based on splits marked in 'SemEval_set' column of dataset
        train_mask = df['SemEval_set'] == 'TRAIN'
        val_mask = df['SemEval_set'] == 'TRIAL'
        test_mask = df['SemEval_set'] == 'TEST'

        # applying boolean masks on split
        train_data_set = df[train_mask]
        val_data_set = df[val_mask]
        test_data_set = df[test_mask]

        # indexing columns of interest
        columns = ["sentence_A","sentence_B","relatedness_score"]
        train_df = train_data_set[columns]
        val_df = val_data_set[columns]
        test_df = test_data_set[columns]

        # preprocessing datasets
        pre = Preprocess(stpwds_file_path)
        self.train_df = pre.perform_preprocessing(train_df, columns_mapping)
        self.val_df = pre.perform_preprocessing(val_df, columns_mapping)
        self.test_df = pre.perform_preprocessing(test_df, columns_mapping)

        logging.info("Reading and preprocessing data from Google Sheets completed...")

    # Rest of the code remains unchanged...
```
In the modified code, `pd.read_csv` has been replaced with the logic to read data from the Google Sheets file, and the file path replaced with the Google Sheets file name. Along with that, code has been added to authorize the Google Sheets API client using the service account key file and fetch the Google Sheets file. Finally, we have modified the code to load data into Panda dataframes using `get_as_dataframe` function from gspread-dataframe package. No changes were made for remaining functions as adjusted logic to read Google Sheets file only affects 'load_data' function.
