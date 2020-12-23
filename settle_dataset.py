import numpy as np
import os
import pickle
import cv2
import pandas as pd

def save_cifar_image(array, path):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.transpose(1,2,0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path, array)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def save_to_image(path='/data_b/lius/cifar-10-batches-py'):
    for i in range(5):
        i+=1
        batch = os.path.join(path, f'data_batch_{i}')
        batch_path = os.path.join(path, batch)
        batch_dict = unpickle(batch_path)
        batch_data = batch_dict[b'data']
        batch_label = batch_dict[b'labels']
        num = batch_data.shape[0]
        batch_data = batch_data.reshape(-1, 3, 32, 32)
        batch_data = batch_data.astype(np.uint8)
        save_dir = os.path.join(path, f'batch_{i}')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for j in range(num):
            print(batch_data[j].shape)
            save_cifar_image(batch_data[j],os.path.join(save_dir,f'{j}.jpg'))
        np.savetxt('label_batch_{}.csv'.format(i),batch_label)

def generate_csv(path='/data_b/lius/cifar-10-batches-py'):
    cifar10 = {'filepath':[], 'label':[]}
    for i in range(1,6):
        data_batch = os.path.join(path, f'batch_{i}')
        label_batch = os.path.join(path, f'label_batch_{i}.csv')
        label_batch = np.loadtxt(label_batch)
        for j in range(10000):
            cur_img_path = os.path.join(data_batch, f'{j}.jpg')
            cifar10['filepath'].append(cur_img_path)
            cifar10['label'].append(label_batch[j])
    df = pd.DataFrame(data=cifar10)
    df.to_csv('./cifar10.csv')

def save_test(path='/data_b/lius/cifar-10-batches-py'):
    batch = os.path.join(path, 'test_batch')
    batch_path = os.path.join(path, batch)
    batch_dict = unpickle(batch_path)
    batch_data = batch_dict[b'data']
    batch_label = batch_dict[b'labels']
    num = batch_data.shape[0]
    batch_data = batch_data.reshape(-1, 3, 32, 32)
    batch_data = batch_data.astype(np.uint8)
    save_dir = os.path.join(path, 'test_batch')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for j in range(num):
        # print(batch_data[j].shape)
        save_cifar_image(batch_data[j], os.path.join(save_dir, f'{j}.jpg'))
        if (j+1) % 1000 == 0:
            print(f'processed {j+1} images')
    np.savetxt('test_batch.csv', batch_label)

def generate_test_csv(path='/data_b/lius/cifar-10-batches-py'):
    cifar10 = {'filepath': [], 'label': []}
    data_batch = os.path.join(path, 'test_batch')
    label_batch = os.path.join(path, 'test_batch.csv')
    label_batch = np.loadtxt(label_batch)
    for j in range(10000):
        cur_img_path = os.path.join(data_batch, f'{j}.jpg')
        cifar10['filepath'].append(cur_img_path)
        cifar10['label'].append(label_batch[j])
        if (j+1) % 3000 == 0:
            print(f'processed {j+1} images')
    df = pd.DataFrame(data=cifar10)
    df.to_csv('./cifar10_test.csv')



save_test()
generate_test_csv()


