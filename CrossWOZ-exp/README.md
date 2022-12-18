# VLS-GPT on CrossWOZ
This is the code and data for CrossWOZ experiments.
## Requirements
* python=3.6
* pytorch=1.2.0
* trasformers=3.5.1
## Data preprocessing
First, you need to extract data from zip files:
```
unzip data.zip
unzip db.zip
```
Then preprocess these files
```
python utils.py
python data_analysis_cross.py
python preprocess_cross.py
```
## Training
### Supervised training over all data in CrossWOZ:
```
bash train.sh $GPU
```
You can change other parameters such as batch size and learning rate in this `.sh` file. For instance, if your GPU has enough memory, you can increase the batch size and decrease the gradient accumulation steps.
### Supervised pre-training with partial labeled data
Our semi-supervised learning is divided into two stages. The first stage is supervised pre-training with part of labeled data and the second stage is semi-supervised training.

During the first stage, to train the generative model, run
```
bash pretrain.sh $GPU $ratio
```
`ratio` means the proportion of labeled data in total data. For example, if `ratio=20`, then 20% of data in CrossWOZ are regarded as labeled data, and the rest are regarded as unlabeled data.

To train the inference model, run
```
bash pretrain_post.sh $GPU $ratio
```
### Semi-supervised training
After supervised pre-training the generative model of corresponding ratio, you can run the classic semi-supervised learning method self-training (Semi-ST)
```
bash train_ST.sh $GPU $ratio
```
After supervised pre-training the generative model and inference model of corresponding ratio, you can run our semi-supervised variational learning method (Semi-VL):
```
bash train_VL.sh $GPU $ratio
```
During Semi-VL, we recommend using two GPUs (for example, GPU=0,1), one for generating the model and one for inferring the model.
## Evaluation 
To test the performance of your model on the test set of CrossWOZ:
```
bash test.sh $GPU $path
```