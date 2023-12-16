import gspread
from torch.utils.data import Dataset
from oauth2client.service_account import ServiceAccountCredentials

"""
Standard Pytorch Dataset class for loading datasets including
the modification to load data from "Combined_Dataset" Google Sheets file.
"""

class STSDataset(Dataset):
    def __init__(
        self,
        scope,
        creds,
        client,
        sheet,
        sent1_tensor,
        sent2_tensor,
        target_tensor,
        sents1_length_tensor,
        sents2_length_tensor,
        raw_sents_1,
        raw_sents_2,
    ):
        """
            initializes  and populates the the length, data and target tensors, and raw texts list from the Combined Dataset Google sheets file. "scope", "creds", "client", "sheet" can be used to authenticate and access the Google Sheets file.
        """
        assert (
            sent1_tensor.size(0)
            == target_tensor.size(0)
            == sent2_tensor.size(0)
            == sents1_length_tensor.size(0)
            == sents2_length_tensor.size(0)
        )

        self.scope = scope
        self.creds = creds
        self.client = client
        self.sheet = sheet
        self.sent1_tensor = sent1_tensor
        self.sent2_tensor = sent2_tensor
        self.target_tensor = target_tensor
        self.sents1_length_tensor = sents1_length_tensor
        self.sents2_length_tensor = sents2_length_tensor
        self.raw_sents_1 = raw_sents_1
        self.raw_sents_2 = raw_sents_2

    def __getitem__(self, index):
        """
            returns the tuple of data tensor, targets, lengths of sequences tensor and raw texts list from the Google Sheets file
        """
        return (
            self.sent1_tensor[index],
            self.sent2_tensor[index],
            self.sents1_length_tensor[index],
            self.sents2_length_tensor[index],
            self.target_tensor[index],
            self.raw_sents_1[index],
            self.raw_sents_2[index],
        )

    def __len__(self):
        """
            returns the length of the data tensor.
        """
        return self.target_tensor.size(0)