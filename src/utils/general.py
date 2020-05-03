"""
General Utilities
"""
import os
import io
import mmap
import torch
import random
import numpy as np
import pandas as pd  # type: ignore
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from argparse import Namespace
import matplotlib.pyplot as plt
import seaborn as sns


def alert():
    from IPython.display import Audio

    wave = np.sin(2 * np.pi * 400 * np.arange(10000 * 0.35) / 10000)
    Audio(wave, rate=10000, autoplay=True)


def plot_train_state(train_state):
    """Plot the train state
    Args:
        train_state (dict): Dict containing train state information
    """

    sns.set(style="darkgrid")

    plot_df = pd.DataFrame(
        {
            "train_acc": train_state["train_accuracies"],
            "val_acc": train_state["val_accuracies"],
        }
    )
    plot_df.index += 1
    num_epochs = len(plot_df)

    fig, ax = plt.subplots(figsize=(10, 7))

    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, num_epochs + 1, 1))
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    axp = sns.lineplot(ax=ax, data=plot_df, legend="full")
    for epoch, train_acc, val_acc in zip(
        range(1, num_epochs + 1), plot_df["train_acc"], plot_df["val_acc"]
    ):
        plt.annotate(
            f"{train_acc:.3f}",
            xy=(epoch, train_acc),
            xytext=(0, 30),
            textcoords="offset points",
            ha="center",
            va="top",
            bbox=dict(boxstyle="square,pad=0.2", alpha=0.5),
            #         arrowprops=dict(arrowstyle = 'simple', connectionstyle='arc3,rad=0'),
        )
        plt.annotate(
            f"{val_acc:.3f}",
            xy=(epoch, val_acc),
            xytext=(0, -30),
            textcoords="offset points",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="square,pad=0.2", fc="orange", alpha=0.5),
            #         arrowprops=dict(arrowstyle = 'simple', connectionstyle='arc3,rad=0'),
        )


def get_misclassified_examples(torch_dataset, split_type, train_state, threshold=0.5):
    torch_dataset.set_split(split_type)
    new_df = torch_dataset._target_df.iloc[
        train_state[f"{split_type}_indexes"][-1].cpu().numpy()
    ]
    new_df.reset_index(drop=True, inplace=True)
    y_pred = (
        (torch.sigmoid(train_state[f"{split_type}_preds"][-1]) > threshold).cpu().long()
    )
    new_df = new_df.assign(pred=pd.Series(y_pred))
    new_df = new_df[new_df.label != new_df.pred][["text", "label", "pred"]]

    return new_df


def analyse_preds(y_pred, y_target, threshold=0.5):
    y_pred = (torch.sigmoid(y_pred) > threshold).cpu().long().numpy()
    y_target = y_target.cpu().numpy()

    conmat = confusion_matrix(y_pred=y_pred, y_true=y_target)
    confusion = pd.DataFrame(
        conmat, index=["NOT", "HS"], columns=["predicted_NOT", "predicted_HS"]
    )
    print("acc = ", accuracy_score(y_pred=y_pred, y_true=y_target))
    print(classification_report(y_pred=y_pred, y_true=y_target, digits=4))
    print(confusion)


def make_train_state():
    d = {
        "train_preds": [],
        "train_indexes": [],
        "train_targets": [],
        "train_accuracies": [],
        "train_f1s": [],
        "train_losses": [],
        "val_preds": [],
        "val_indexes": [],
        "val_targets": [],
        "val_accuracies": [],
        "val_f1s": [],
        "val_losses": [],
        "test_preds": [],
        "test_indexes": [],
        "test_targets": [],
        "test_accuracies": [],
        "test_f1s": [],
        "test_losses": [],
        "batch_preds": [],
        "batch_targets": [],
        "batch_indexes": [],
        "epoch_index": 0,
        # "save_path": ''
    }
    return dict(d)


def compute_accuracy(y_pred, y_target):
    y_target = y_target.cpu()
    y_pred_indices = (torch.sigmoid(y_pred) > 0.5).cpu().long()
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def set_seed_everywhere(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe_tensor(x):
    """
    Prints information about a given tensor
    """
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))


class DefaultFilePaths:
    """
    Helper class that stores the location of datafiles, embeddings, etc.
    Must be set up for your local machine. Default configuration is for the maintainer's
    personal machine.
    """

    def __init__(self, location="local"):
        if location == "local":
            self.PREFIX = "/Users/cozek/Documents/MTech/4th Sem/OffensEval/data"
            self.glove = "/Users/cozek/Documents/MTech/3rd Sem/Project/glove.twitter.27B/glove.twitter.27B.200d.txt"
            self.fasttext_bin = (
                "/Users/cozek/Documents/MTech/3rd Sem/Project/cc.en.300.bin"
            )
            self.bert_uncased_large = (
                "/Users/cozek/Documents/MTech/4th Sem/wwm_uncased_L-24_H-1024_A-16/"
            )
            self.gpt_2 = "/Users/cozek/Documents/MTech/4th Sem/gpt_2/"
            self.offeval_data = {
                "en": {
                    "task_a": self.PREFIX
                    + "/OffenseEval2020Data/English/task_a_distant.tsv",
                    "task_b": self.PREFIX
                    + "/OffenseEval2020Data/English/task_b_distant.tsv",
                    "task_c": self.PREFIX
                    + "/OffenseEval2020Data/English/task_c_distant.tsv",
                },
                "en_presplit": self.PREFIX
                + "/OffenseEval2020Data/English/task_a_split.csv",
                "en_presplit_lite": self.PREFIX
                + "/OffenseEval2020Data/English/task_a_split_lite.csv",
                "en_presplit_tiny": self.PREFIX
                + "/OffenseEval2020Data/English/task_a_split_tiny.csv",
            }
            self.hasoc_data = {
                "en": {
                    "train": self.PREFIX + "/hasoc_data/en/english_dataset.tsv",
                    "test": self.PREFIX + "/hasoc_data/gold/hasoc2019_en_test-2919.tsv",
                },
                "en_presplit_task_a": self.PREFIX
                + "/hasoc_data/en/en_presplit_task_a.csv",
                "en_presplit_task_a_lite": self.PREFIX
                + "/hasoc_data/en/en_presplit_task_a_tiny.csv",
            }
        elif location == "server":
            self.PREFIX = "/home/kaushik.das/OffensEval2020/data"
            self.glove = "/home/kaushik.das/embeddings/glove.twitter.27B.200d.txt"
            self.fasttext_bin = "/home/kaushik.das/embeddings/crawl-300d-2M-subword.bin"
            self.bert_uncased_large = (
                "/home/kaushik.das/pytorch_transformers/bert_uncased/"
            )
            self.memotion = {
                'loc' : self.PREFIX + '/memotion_dataset_7k/',
                'task_a_advprop_df': self.PREFIX + 'memotion_dataset_7k/images_advprop_df_task_a.pickle',
                'task_a_simple_df': self.PREFIX + 'memotion_dataset_7k/images_simple_df_task_a.pickle',

            }
            self.gpt_2 = "/home/kaushik.das/pytorch_transformers/gpt2/"
            self.distilgpt2 = "/home/kaushik.das/pytorch_transformers/distilgpt2/"
            self.model_storage = "/home/kaushik.das/OffensEval2020/saved_models/"
            self.trac_data = {
                "en_dev": self.PREFIX + "/TRAC/eng/trac2_eng_dev.csv",
                "en_train": self.PREFIX + "/TRAC/eng/trac2_eng_train.csv",
                "en_task_a_dataframe": self.PREFIX + "/TRAC/eng/trac2_eng_task_a_df.csv",
                "en_task_b_dataframe": self.PREFIX + "/TRAC/eng/trac2_eng_task_b_df.csv",


                "hin_dev": self.PREFIX + "/TRAC/hin/trac2_hin_dev.csv",
                "hin_train": self.PREFIX + "/TRAC/hin/trac2_hin_train.csv",
                "iben_dev": self.PREFIX + "/TRAC/iben/trac2_iben_dev.csv",
                "iben_train": self.PREFIX + "/TRAC/iben/trac2_iben_train.csv",
            }
            self.offeval_data = {
                "en": {
                    "task_a": self.PREFIX
                    + "/OffenseEval2020Data/English/task_a_distant.tsv",
                    "task_b": self.PREFIX
                    + "/OffenseEval2020Data/English/task_b_distant.tsv",
                    "task_c": self.PREFIX
                    + "/OffenseEval2020Data/English/task_c_distant_ann.tsv",
                },
                # TASK C
                "en_task_c_presplit_final": self.PREFIX
                + "/OffenseEval2020Data/English/offeval2020_task_c_en_presplit.csv",
                "en_task_c_presplit_lite": self.PREFIX
                + "/OffenseEval2020Data/English/en_task_c_presplit_lite.csv",
                "en_task_c_presplit_full": self.PREFIX
                + "/OffenseEval2020Data/English/en_task_c_presplit_full.csv",
                # TASK B
                "en_task_b_presplit_lite": self.PREFIX
                + "/OffenseEval2020Data/English/en_task_b_presplit_lite.csv",
                "en_task_b_presplit_full": self.PREFIX
                + "/OffenseEval2020Data/English/en_task_b_presplit_full.csv",
                "en_public_test_b": self.PREFIX  # testset
                + "/OffenseEval2020Data/English/task_b_test/test_b_tweets.tsv",
                # TASK A
                "en_public_test_a": self.PREFIX  # testset
                + "/OffenseEval2020Data/English/public_data_A/test_a_tweets.tsv",
                "en_presplit_full": self.PREFIX
                + "/OffenseEval2020Data/English/task_a_split_full.csv",
                "en_presplit_lite": self.PREFIX
                + "/OffenseEval2020Data/English/task_a_split_lite.csv",
                "en_presplit_tiny": self.PREFIX
                + "/OffenseEval2020Data/English/task_a_split_tiny.csv",
                "en_presplit_tiny_fixed": self.PREFIX
                + "/OffenseEval2020Data/English/task_a_split_tiny_fixed.csv",
                "en_presplit_lite_fixed": self.PREFIX
                + "/OffenseEval2020Data/English/task_a_split_lite_fixed.csv",
                "en_presplit_full_fixed": self.PREFIX
                + "/OffenseEval2020Data/English/task_a_split_full_fixed.csv",
                # std <= 0.3
                "en_safe_presplit_full": self.PREFIX
                + "/OffenseEval2020Data/English/task_a_split_full_safe.csv",
                # std <= 0.2
                "en_verysafe_presplit_tiny": self.PREFIX
                + "/OffenseEval2020Data/English/task_a_split_tiny_verysafe.csv",
                "en_verysafe_presplit_full": self.PREFIX
                + "/OffenseEval2020Data/English/task_a_split_full_verysafe.csv",
            }

            self.hasoc_data = {
                "en": {
                    "train": self.PREFIX + "/hasoc_data/en/english_dataset.tsv",
                    "test": self.PREFIX + "/hasoc_data/gold/hasoc2019_en_test-2919.tsv",
                },
                "en_presplit_task_a": self.PREFIX
                + "/hasoc_data/en/en_presplit_task_a.csv",
                "en_presplit_task_a_lite": self.PREFIX
                + "/hasoc_data/en/en_presplit_task_a_tiny.csv",
            }


if __name__ == "__main__":
    d = DefaultFilePaths()
