"""
Utils for OffensEval 2020 dataset
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
    data_df = pd.read_csv(path, sep="\t", quoting=3)
    # remove errenous space character in column name in some files
    data_df.columns = list(map(lambda c: c.strip(), data_df.columns))

    return data_df


def labeller(df: pd.DataFrame, threshold: float, task: str, drop_cols: bool):
    """Adds a label to the samples in the given DataFrame
    Args:
        df (pd.DataFrame): A dataframe containing the samples their given confidence
        as df.text and df.average respectively
        threshold (float):  Probability to label a sample as positive
        task: one of 'a','b','c'
        drop_cols: drops some columns that are not necessary 
    Returns:
        df (pd.DataFrame): with a added column that labels of each sample respectively
        label_dict (dict): The labels and their corresponding integer values 
    """
    task = task.lower()

    assert isinstance(df, pd.DataFrame)
    assert 0.0 <= threshold <= 1.0
    assert task in ["a", "b", "c"]
    assert isinstance(drop_cols, bool)

    if task in ["a", "b"]:
        df["label"] = df.average >= 0.5
        df["label"] = df["label"].astype(int)
    elif task == "c":
        cols = {"average_ind": 0, "average_grp": 1, "average_oth": 2}
        df["label"] = df[list(cols.keys())].idxmax(axis=1)
        df["label"] = df["label"].apply(lambda x: cols[x])

    if drop_cols:
        df = df[["id", "text", "label"]]

    label_dict = {}
    if task == "a":
        label_dict["OFF"] = 1
        label_dict["NOT"] = 0
    elif task == "b": #bug
        label_dict["UNT"] = 1 
        label_dict["TIN"] = 0
    elif task == "c":
        label_dict["IND"] = 0
        label_dict["GRP"] = 1
        label_dict["OTH"] = 2

    return df, label_dict
