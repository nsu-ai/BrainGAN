import os
import PIL
import zipfile
import numpy as np
import h5py
import requests
import matplotlib.pyplot as plt
import shutil

if not os.path.exists('dataset'):
    os.mkdir('dataset')

    dataset_links = [
        'https://ndownloader.figshare.com/files/3381290',
        'https://ndownloader.figshare.com/files/3381293',
        'https://ndownloader.figshare.com/files/3381296',
        'https://ndownloader.figshare.com/files/3381302'
    ]

    dataset_names = [
        'dataset_part1',
        'dataset_part2',
        'dataset_part3',
        'dataset_part4'
    ]

    for i, link in enumerate(dataset_links):
        print("Downloading {}".format(dataset_names[i]))
        with requests.get(link, stream=True) as r:
            r.raise_for_status()
            with open('dataset/' + dataset_names[i], 'wb') as f:
                shutil.copyfileobj(r.raw, f)

        with zipfile.ZipFile('dataset/' + dataset_names[i]) as zf:
            zf.extractall('dataset')
        
        os.remove('dataset/' + dataset_names[i])

    print("Extracted .mat files...")

if not os.path.exists('data'):
    os.mkdir('data')

    os.mkdir('data/Meningioma')
    os.mkdir('data/Glioma')
    os.mkdir('data/Pituitary')

    filename = None

    for filename in range(1, 3065):
        with h5py.File('dataset/{}.mat'.format(filename), 'r') as f:
            img = f['cjdata']['image']
            label = f['cjdata']['label'][0][0]

            img = np.array(img, dtype=np.float32)

            path = None

            if label == 1 or label == '1':
                path = 'data/Meningioma/{}.png'.format(filename)
            elif label == 2 or label == '2':
                path = 'data/Glioma/{}.png'.format(filename)
            elif label == 3 or label == '3':
                path = 'data/Pituitary/{}.png'.format(filename)
            else:
                print("Wrong label found (Label Noise). Discarding sample")   
                continue  
            
            plt.axis('off')
            plt.imsave(path, img, cmap='gray')
        
    print("{} files successfully saved as images under data directory with label names as sub-folders".format(filename))

else:
    flag = 0

    for dir in ['data/Meningioma', 'data/Glioma', 'data/Pituitary']:
        if not os.path.exists(dir):
            os.mkdir(dir)
            flag = 1
    
    if flag == 1:
        filename = None

        for filename in range(1, 3065):
            with h5py.File('dataset/{}.mat'.format(filename), 'r') as f:
                img = f['cjdata']['image']
                label = f['cjdata']['label'][0][0]

                img = np.array(img, dtype=np.float32)

                path = None

                if label == 1 or label == '1':
                    path = 'data/Meningioma/{}.png'.format(filename)
                elif label == 2 or label == '2':
                    path = 'data/Glioma/{}.png'.format(filename)
                elif label == 3 or label == '3':
                    path = 'data/Pituitary/{}.png'.format(filename)
                else:
                    print("Wrong label found (Label Noise). Discarding sample")   
                    continue  
                
                plt.axis('off')
                plt.imsave(path, img, cmap='gray')
            
        print("{} files successfully saved as images under data directory with label names as sub-folders".format(filename))

exit()