# VLS-GPT on MultiWOZ2.1
This is the code and data for MultiWOZ2.1 experiments.
## Requirements
* python=3.6
* pytorch=1.2.0
* trasformers=3.5.1
## Data preprocessing
First, you need to unzip the data and database files.
```
unzip data.zip
unzip db.zip
```
You can **directly use the preprocessed data** in [data/multi-woz-2.1-processed](./data/multi-woz-2.1-processed/) or preprocess the data from scratch with following commands:
```
python data_analysis.py
python preprocess.py
```
## Training
### Supervised training over all data in MultiWOZ2.1
```
bash train.sh $GPU
```
You can change other parameters such as batch size and learning rate in this `.sh` file.

### Supervised pre-training with partial labeled data

Our semi-supervised learning is divided into two stages. The first stage is supervised pre-training on the labeled data and the second stage is semi-supervised training on the mixture of labeled and unlabeled data.

During the first stage, to train the generative model, run
```
bash pretrain.sh $GPU $ratio
```
`ratio` means the proportion of labeled data in total data. For example, if `ratio=20`, then 20% of data in MultiWOZ are regarded as labeled data, and the rest are regarded as unlabeled data.

To train the inference model, run
```
bash pretrain_post.sh $GPU $ratio
```

### Semi-supervised training
After supervised pre-training the generative model of corresponding ratio, you can run the classic semi-supervised learning method self-training (Semi-ST)
```
bash train_ST.sh $GPU $ratio
```
After supervised pre-training, you can run our semi-supervised variational learning method (Semi-VL):
```
bash train_VL.sh $GPU $ratio
```
During Semi-VL, we recommend using two GPUs (for example, GPU=0,1), one for the generative model and one for the inference the model.

### Data augmentation
To train the system with augmented data through back-translation on low-resource scenario, first you need to prepare the data
```
python prepare_aug_data.py
``` 
Then you can train the system at a certain proportion of labeled data
```
bash pretrain_aug.sh $GPU $ratio
```

## Evaluation 
To test the performance of your model on the test set of MultiWOZ2.1:
```
bash test.sh $GPU $path
```