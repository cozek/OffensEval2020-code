"""
Utilities for creating the dataset for 

"""
from typing import Callable
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
import collections
import pandas as pd
import numpy as np
import string
import torch
import nltk

class GPT2Preprocessor():
    def __init__(self,transformer_tokenizer,sentence_detector):
        self.transformer_tokenizer = transformer_tokenizer
        self.sentence_detector = sentence_detector
        
    def add_eos_tokens(self, text):
        eos_token = ' ' + self.transformer_tokenizer.eos_token + ' '
        sentences = self.sentence_detector.tokenize(text)
        eos_added_text  = eos_token.join(sentences) + ' ' + self.transformer_tokenizer.eos_token
        return eos_added_text 

class Vectorizer():
    def __init__(self,tokenizer: Callable, max_seq_len: int):
        """
        Args:
            tokenizer (Callable): transformer tokenizer
            max_seq_len (int): Maximum sequence lenght 
        """
        self.tokenizer = tokenizer
        self._max_seq_len = max_seq_len

    def vectorize(self,text :str):
        sequence = \
            self.tokenizer.prepare_for_tokenization(text,add_prefix_space=True)
        indices = self.tokenizer.encode(sequence)
        
        out_vector = np.zeros(self._max_seq_len, dtype=np.int64)
        out_vector[: len(indices)] = indices
        # max len is restricted to 1024
        return out_vector[:min(self._max_seq_len,1024)]        

class HateDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, tokenizer: Callable, max_len:int=None):
        """
        Args:
            data_df (pandas.DataFrame): df containing the labels and text
            tokenizer (tokenizer module for the transformer)
        """
        self.data_df = data_df
        self.tokenizer = tokenizer

        # measure_len = lambda context: len(context.split(" "))
        # self._max_seq_length = max(map(measure_len, data_df.text)) + 2
        if max_len == None:
            self._max_seq_length = self._get_max_len(data_df,tokenizer)
        else:
            self._max_seq_length = max_len

        self.train_df = self.data_df[self.data_df.split == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.data_df[self.data_df.split == 'val']
        self.val_size = len(self.val_df)

        self.test_df = self.data_df[self.data_df.split == 'test']
        self.test_size = len(self.test_df)


        self._vectorizer = Vectorizer(tokenizer, self._max_seq_length)


        self._lookup_dict = {
            'train': (self.train_df, self.train_size),
            'val': (self.val_df, self.val_size),
            'test': (self.test_df, self.test_size)
        }

        self.set_split('train')

        class_counts = data_df.label.value_counts().to_dict()
         #sorted on the basis of class label,eg, 0,1,2..
        cts = sorted([(lbl,cts) for lbl,cts in class_counts.items()], key=lambda x: x[0])
        freq = [ x[1] for x in cts ]
        # print(freq,cts)
        self.class_weights = 1.0/ torch.tensor(freq, dtype=torch.float32)
    
    def _get_max_len(self,data_df: pd.DataFrame, tokenizer: Callable):
        prep_func = lambda x: self.tokenizer.prepare_for_tokenization(x,add_prefix_space=True)
        len_func = lambda x: len(prep_func(x))
        max_len = data_df.text.map(len_func).max() 
        return max_len

        # max_len = 0
        # for seq in data_df['text']:
        #     temp = tokenizer.prepare_for_tokenization(seq,add_prefix_space=True)
        #     tokenized_seq = tokenizer.tokenize(temp)
        #     if len(tokenized_seq) > max_len:
        #         max_len = len(tokenized_seq)
        # return max_len

        

    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]
    
    def __len__(self):
        return self._target_size
    
    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets
        
        Args:
            index (int): the index to the data point 
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        sequence = self._vectorizer.vectorize(row.text)

        label = row.label
        return {'x_data': sequence,
                'x_index': index,
                'y_target': label}
    
    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
        
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size

class TracDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame, tokenizer: Callable):
        """
        Args:
            data_df (pandas.DataFrame): df containing the labels and text
            tokenizer (tokenizer module for the transformer)
        """
        self.data_df = data_df
        self.tokenizer = tokenizer

        # measure_len = lambda context: len(context.split(" "))
        # self._max_seq_length = max(map(measure_len, data_df.text)) + 2
        self._max_seq_length = self._get_max_len(data_df,tokenizer)

        self.train_df = self.data_df[self.data_df.split == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.data_df[self.data_df.split == 'dev']
        self.val_size = len(self.val_df)

        self.test_df = self.data_df[self.data_df.split == 'test']
        self.test_size = len(self.test_df)


        self._vectorizer = Vectorizer(tokenizer, self._max_seq_length)
        

        self._lookup_dict = {
            'train': (self.train_df, self.train_size),
            'val': (self.val_df, self.val_size),
            'test': (self.test_df, self.test_size)
        }

        self.set_split('train')

        class_counts = data_df.label.value_counts().to_dict()
         #sorted on the basis of class label,eg, 0,1,2..
        cts = sorted([(lbl,cts) for lbl,cts in class_counts.items()], key=lambda x: x[0])
        freq = [ x[1] for x in cts ]
        # print(freq,cts)
        self.class_weights = 1.0/ torch.tensor(freq, dtype=torch.float32)
    
    def _get_max_len(self,data_df: pd.DataFrame, tokenizer: Callable):
        max_len = 0
        for seq in data_df['text']:
            temp = tokenizer.prepare_for_tokenization(seq,add_prefix_space=True)
            tokenized_seq = tokenizer.tokenize(temp)
            if len(tokenized_seq) > max_len:
                max_len = len(tokenized_seq)
        return max_len

        

    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]
    
    def __len__(self):
        return self._target_size
    
    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets
        
        Args:
            index (int): the index to the data point 
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        sequence = self._vectorizer.vectorize(row.text)

        label = row.label
        return {'x_data': sequence,
                'x_index': index,
                'y_target': label}
    
    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
        
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size

def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=False, device="cpu", pinned_memory = False, n_workers = 0): 
    """
    A generator function which wraps the PyTorch DataLoader. It will 
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last,
                            pin_memory= pinned_memory,
                            num_workers = n_workers,
                            )

    for data_dict in dataloader:
        out_data_dict = {}
        # print(data_dict.items())
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device, non_blocking= (True if pinned_memory else False) )
        yield out_data_dict