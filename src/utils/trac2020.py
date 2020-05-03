"""
Utils for TRAC 2020 dataset
"""
from argparse import Namespace
from tqdm import tqdm
import pandas as pd  # type: ignore
import numpy as np
import random
import torch
import mmap
import io
import os

def load_dataset(path: str):
    """Loads OffensEval 2020 dataset
    Args:
        path (str): full path of a task_*_distant.tsv file
            provided by the organiser
    Returns:
        pandas.DataFrame containing the data
    """
    data_df = pd.read_csv(path)
    # remove errenous space character in column name in some files
    data_df.columns = list(map(lambda c: c.strip(), data_df.columns))

    return data_df

def chunk_sent(text1:str, n_words_per_chunk:int, n_prev:int):
    """
    Chunks sentences into chunks having n_words_per_chunk
    using a window that considers the last n_prev words of the previous chunk
    
    >>> some_text = "w1 w2 w3 w5. w6 w7 w8"
    >>> chunk_sent(some_text, 3,1)
    ['w1 w2 w3', 'w3 w5. w6', 'w6 w7 w8']
    """
    alpha = n_words_per_chunk - n_prev
    l_total = []
    l_parcial = []
    if len(text1.split())//alpha >0:
        n = len(text1.split())//alpha
    else: 
        n = 1
    for w in range(n):
        if w == 0:
            l_parcial = text1.split()[:n_words_per_chunk]
            l_total.append(
                " ".join(l_parcial)
            )
        else:
            l_parcial = text1.split()[w*alpha:w*alpha + n_words_per_chunk]
            l_total.append(" ".join(l_parcial))
    return l_total

def get_label_dict(task:str):
    """Returns the label dict for the task
    Args:
        task: one of 'a' , 'b'
    """

    assert isinstance(task,str)

    task == task.lower()
    if task == 'a':
        task_a_label_dict = {'NAG':0, 'CAG':1, 'OAG':2}
        return task_a_label_dict
    elif task  == 'b':
        task_b_label_dict = {'NGEN':0, 'GEN':1}
        return task_b_label_dict
    else:
        raise ValueError("Must be on of ['a','b'] !")
