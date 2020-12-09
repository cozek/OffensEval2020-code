# OffensEval2020_submission 
[OffensEval 2020](https://sites.google.com/site/offensevalsharedtask/home) Models code for Team KAFK

Paper Link: https://www.aclweb.org/anthology/2020.semeval-1.267.pdf

Please find the notebooks for the system code used for each task  in the `notebooks` directory.
They should work out of the box in Google Colab. However, to fully replicate our work you will need the exact hyperparmeters
from the original paper and the full dataset which might not be possible in Colab. 

We have provided small subset of the dataset for each task in the `data` folder to use with the abovementioned notebooks. Please cite their work if you used the data in your work. The citation is provided below. Also, if you want to use the full dataset, kindly create DataFrames out of them in the same manner as used in the notebooks.


Credits:

- RAdam : https://github.com/LiyuanLucasLiu/RAdam
- LookAhead: https://github.com/lonePatient/lookahead_pytorch
- Transformers: https://github.com/huggingface/transformers


If you found our paper/scripts useful cite:
```
@inproceedings{das2020kafk,
  title={KAFK at SemEval-2020 Task 12: Checkpoint Ensemble of Transformers for Hate Speech Classification},
  author={Das, Kaushik Amar and Baruah, Arup and Barbhuiya, Ferdous Ahmed and Dey, Kuntal},
  booktitle={Proceedings of the Fourteenth Workshop on Semantic Evaluation},
  pages={2023--2029},
  year={2020}
}
```

If you used the data please cite
```
@inproceedings{rosenthal2020,
    title={{A Large-Scale Semi-Supervised Dataset for Offensive Language Identification}},
    author={Rosenthal, Sara and Atanasova, Pepa and Karadzhov, Georgi and Zampieri, Marcos and Nakov, Preslav},
    year={2020},
    booktitle={arxiv}
 }
```
