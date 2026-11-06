# TDM4Rec

# Overview
TDM4Rec is a Mamba-based architecture, TDM4Rec first integrates item and time sequence embeddings to construct time-aware interaction sequence representations through the embedding layer. These representations are then processed by $L$ stacked Denoising-Mamba layers, each consisting of a frequency-domain filtering layer, a Mamba layer, and a feed-forward network (FFN). Finally, the prediction layer generates recommendation scores for candidate items.

![Framework](Images/model.jpg)


# Datasets
We utilize six benchmark datasets to evaluate TDM4Rec, all of which are obtained from [RecBole](https://github.com/RUCAIBox/RecSysDatasets).
* Amazon Beauty, Video, Toys and Sports are four representative sub-datasets acquired from the [Amazon platform](https://jmcauley.ucsd.edu/data/amazon/), which includes abundant user interaction records. These datasets are characterized by relatively short average interaction sequences.
* Steam is a video game review dataset collected from the online gaming platform Steam.
* ML-1M is a movie rating dataset collected from the [MovieLens website](https://grouplens.org/datasets/movielens/), characterized by relatively long user-movie rating interaction sequences.

All dataset files can be obtained from the cloud storage [BaiDu Cloud Drive (Password: e272)](https://pan.baidu.com/s/1p51sWMgVFbAaHQmL4aD_-g#list/path=%2F).


# Baselines
The detailed descriptions of the baseline methods used for comparison with our TDM4Rec are given below:

* GRU4Rec: captures sequential dependencies in interaction sequences by leveraging GRU.

* BERT4Rec: acquires global context by implementing a bi-directional Transformer under the paradigm of BERT. 

* SASRec: extracts sequential characteristics relying on a unidirectional Transformer.

* TiSASRec: the first method to incorporate temporal information into Transformer-based SR by constructing a time difference matrix and integrating it into the attention mechanism.

* FMLP-Rec: refines sequence representation via learnable filters, which is the pioneer exploration of frequency-based filtering technique in SR.

* FreqRec: jointly captures intra- and inter-sequential dynamics by integrating a learnable frequency-domain MLP with self-attention, and further improves performance by incorporating a frequency-domain consistency loss. 

* Mamba4Rec: captures sequential dependencies via the selective SSM for efficient sequence modeling, which is the first exploration of Mamba architecture in SR.

* ECHOMamba4Rec: employs bi-directional Mamba to enhance global modeling with frequency-domain filtering technique.

* SIGMA: strengthens global and short-sequence modeling by adopting a specific bi-directional Mamba coupled with variant GRU.

* TiM4Rec: incorporates temporal information into Mamba through a time-aware structured masked matrix while preserving linear computational complexity.




# Environment Setting
 ```
conda create -n TDM4Rec python=3.8
conda activate TDM4Rec
 ```
The required environment settings are detailed in the `requirements.txt` file.


# Quick-Start
## How to run
Just run it directly：
 ```
python main.py  
 ```

Example for Beauty
 ```
config = Config(model=FDM4Rec, config_file_list=['config/config.yaml'])
 ```

## Acknowledgements
Our code implementation is based on the [RecBole](https://github.com/RUCAIBox/RecBole) and [Pytorch](https://github.com/pytorch/pytorch) frameworks.
