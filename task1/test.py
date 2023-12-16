```python
import torch
from scipy import stats
import logging
from torch import nn
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import mean_squared_error
from gspread_dataframe import get_as_dataframe # Added for data loading from Google Sheets

logging.basicConfig(level=logging.INFO)

def evaluate_test_set(model, data_loader, config_dict):
    """
    Evaluates the model performance on dev data
    """
    logging.info("Evaluating accuracy on test set")

    # Implementation to handle "Combined_Dataset" Google Sheets data format
    predictions = list()
    truths = list()
    total_loss = 0
    device = config_dict["device"]
    criterion = nn.MSELoss()

    for i, data_loader in enumerate(data_loader['test']):
        
        # updated to load data from Combined dataset from Google Sheets
        sent1_batch, sent2_batch, sent1_len, sent2_len, targets = get_as_dataframe(data_loader[0:5])

        pred, A_1, A_2 = model(sent1_batch.to(device), sent2_batch.to(device), sent1_len, sent2_len)
        pred = torch.squeeze(pred)

        loss = criterion(pred.to(device), targets.float().to(device))

        predictions += list(pred.detach().cpu().numpy())
        truths += list(targets.numpy())
        total_loss += loss
    
    # computing accuracy using sklearn's function
    # Using Pearson Coefficient as evaluation metric to store final evaluation result
    acc, p_value = stats.pearsonr(truths, predictions)
    
    print("Accuracy: {} Test Loss: {}".format(acc, torch.mean(total_loss.float())))
```
Changed the code in `evaluate_test_set` function to ensure it can handle the data format of the "Combined_Dataset" Google Sheets file. Added code to load the Google Sheets file into Pandas dataframe using `get_as_dataframe` function from gspread-dataframe package. The loaded dataframe is then used to feed the data to the model for testing.