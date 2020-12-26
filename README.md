# Report of CVML Test: Noisy Label

## Shuai Liu

**Code repository**: https://github.com/1170500804/NoisyLabelTest.git

## 1. Introduction

The language and framework that I use to implement this model is Python3 and Pytorch. The whole project is composited with 6 python files: `backbone.py, data.py, evaluate.py, loss.py, main.py, settle_datset.py`. The following of this section is the description of these files and their functions.

### main.py

Run this file to train the model. To run the improved model, 

```shell
python3 main.py -a 0.1 -b 1 --eta 0.6 --lr 0.1 --batch-size 512 --aug
```

To run the baseline:

```shell
python3 main.py --baseline --eta 0.6 --lr 0.1 --batch-size 512 --aug
```

To evaluate the trained model:

```shell
python3 main.py --eval --resume [path to saved model]
```



### Backbone.py

This file implements the backbone, which is an 8-layer neural network, specified in the problem description.

### Data.py

This file implements the interface that loads data during training. The data augmentation method I use is *Random Crop* and *Random Horizontal Flip.*

### evaluate.py

This file implements the evaluate method.

### loss.py

This file implements the reversed cross entropy with A=-4

### settle_dataset.py

This file transfers format of the cifar-10: pickle object to jpg files.



## Result

