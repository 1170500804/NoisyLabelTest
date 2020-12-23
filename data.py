import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from PIL import Image

def generate_noisy_label(clean_labels, eta=0.6):
    total_num = len(clean_labels)
    noisy_num = round(eta * total_num)
    noisy_index = sorted(np.random.choice(total_num,noisy_num, replace=False))
    noisy_addition = np.random.choice(8, noisy_num)+1
    noisy_addition = noisy_addition.asarray(np.long)
    clean_labels[noisy_index] = clean_labels[noisy_index] + noisy_addition
    noisy_labels = clean_labels % 10
    print(noisy_labels.dtype)
    return noisy_labels

class cifar10(Dataset):
    def __init__(self, transform, path='/data_b/lius/cifar-10-batches-py', eta=0.6):
        self.transform = transform
        self.path = path
        csv_path = os.path.join(self.path, 'cifar10.csv')
        self.cifar10_df = pd.read_csv(csv_path)
        noisy_label = generate_noisy_label(np.array(self.cifar10_df['label'], dtype=np.int8), eta=eta)
        self.cifar10_df['noisy'] = noisy_label
        print(f'generated noisy labels for {len(noisy_label)} instances')

    def __getitem__(self, idx):
        im_info = self.cifar10_df.loc[idx,['filepath','noisy']]
        im_path = im_info['filepath']
        im_label = im_info['noisy']
        im = Image.open(im_path)
        im = self.transform(im)
        return im, im_label

    def __len__(self):
        return len(self.cifar10_df)
