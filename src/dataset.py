import torch
from torch.utils.data import Dataset
from sklearn.datasets import fetch_20newsgroups
from transformers import BertTokenizer

class NewsDataset(Dataset):
    def __init__(self, model_max_length=128):
        self.newsgroups_data = fetch_20newsgroups(subset='all', shuffle=False,)
        self.texts = self.newsgroups_data.data
        self.labels = self.newsgroups_data.target
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                       truncation_side='right',
                                                       padding_side='right',
                                                       model_max_length=model_max_length,
                                                       return_tensors='pt', )
        self.tokenized_data = [self.tokenizer(self.texts[i],
                                              return_tensors='pt',
                                              truncation=True,
                                              padding='max_length')
                               for i in range(len(self.texts))]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tokenized_data[idx], self.labels[idx]

