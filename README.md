# TDM4Rec

# Overview
TDM4Rec is a Mamba-based architecture, TDM4Rec first integrates item and time sequence embeddings to construct time-aware interaction sequence representations through the embedding layer. These representations are then processed by $L$ stacked Denoising-Mamba layers, each consisting of a frequency-domain filtering layer, a Mamba layer, and a feed-forward network (FFN). Finally, the prediction layer generates recommendation scores for candidate items.

![Framework](Images/model.jpg)


# Datasets
We utilize six benchmark datasets to evaluate our TDM4Rec, all of which can be downloaded in [BaiDu Cloud Drive (Password: e272]([https://drive.google.com/drive/folders/1Ir0nVoC_1flw3zW9N_ANck_XaGTvCNTa](https://pan.baidu.com/s/1p51sWMgVFbAaHQmL4aD_-g#list/path=%2F](https://pan.baidu.com/s/1p51sWMgVFbAaHQmL4aD_-g#list/path=%2F)): 
* Amazon Beauty, Sports are two representative sub-datasets gathered from Amazon dataset, which contains a series of product reviews crawled from Amazon.com. They are split by the top-level product categories on Amazon, we adopt the “Beauty”, “Sports and Outdoors (Sports)” categories.
* ML-1M is a large and dense dataset with long item sequences, which collected from the movie recommendation site MovieLens. 
* Yelp is a large business recommendation dataset. We treat the transaction records after $\text{January}$ $1 ^ {st}$,  $2019$.
* Note that the `*_same_target.npy` files in [Google Drive link](https://drive.google.com/drive/folders/1Ir0nVoC_1flw3zW9N_ANck_XaGTvCNTa) for the four datasets are utilized for training DuoRec, FEARec and our MSTRec, both of which incorporate contrastive learning.


## Datasets
This work utilizes the following five datasets, among which Amazon-Beauty, Amazon-Video, Amazon-Toys, Amazon-Sports, Steam and MovieLens-1M. All datasets are provided by [RecBole](https://github.com/RUCAIBox/RecSysDatasets).

All dataset files can be obtained from the cloud storage [BaiDu Cloud Drive (Password: e272)](https://pan.baidu.com/s/1p51sWMgVFbAaHQmL4aD_-g#list/path=%2F).






## Run
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
