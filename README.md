# Description

This repository contains PyTorch implementations of different neural network architectures to classify the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).


```python
import sys

sys.path.append("..")
from utils import utils
from utils import plots
import matplotlib.pyplot as plt

BATCH_SIZE = 256
```


```python
feature_data_raw, label_data_raw = utils.get_raw_data()
features_train, targets_train, features_test, targets_test = utils.get_partitioned_data(feature_data_raw, label_data_raw)
train_loader, test_loader = utils.get_data_loader(features_train, targets_train, features_test, targets_test, batch_size = BATCH_SIZE)
```


```python
plots.inspect_data(train_loader)
```


    
![png](README_files/README_3_0.png)
    

