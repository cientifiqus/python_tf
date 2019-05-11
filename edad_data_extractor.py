import glob
import os
import numpy as np
from PIL import Image


def extract_features(parent_dir,file_ext="*.jpg"):
    age_imgs = []
    labels = []
    for sub_dir in os.listdir(parent_dir):
        print(sub_dir)
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            img = Image.open(fn).convert('L')
            if (img.size[0] >= 90):
                born_year = fn.split('\\')[2].split('_')[1].split('-')[0]
                photo_year = fn.split('\\')[2].split('_')[2].split('.')[0]
                age=int(photo_year)-int(born_year)
                img=img.resize((200,200),Image.ANTIALIAS)
                im=np.array(img)
                age_imgs.append(im)
                labels.append(age)   
    age_imgs = np.asarray(age_imgs).reshape(len(age_imgs),200,200,1)
    return age_imgs, np.array(labels,dtype = np.int)

parent_dir = 'wiki_crop'
features,labels = extract_features(parent_dir)

# Si se requiere se hace alguno de los siguientes procesos.
# Mesclar aleatoriamente la base de datos
x1, y1 = shuffle(features, labels)

# O Separarla en entrenamieto y prueba (o si es necesario en validación)
samples=y1.size
y1=y1.reshape((samples,1))

offset = int(x1.shape[0] * 0.80)
X_train, Y_train = x1[:offset], y1[:offset]
X_test, Y_test = x1[offset:], y1[offset:]
Y_test = np.array(Y_test)
Y_train = np.array(Y_train)
# Se puede adicionar el proceso que requieran para su base de datos

# Este sería el proceso para guardar en un formato h5 los datos
with h5py.File('train_dataset.h5','w') as h5data:
    h5data.create_dataset('train_set_x',data=X_train)
    h5data.create_dataset('train_set_y',data=Y_train)
with h5py.File('test_dataset.h5','w') as h5data:
    h5data.create_dataset('test_set_x',data=X_test)
    h5data.create_dataset('test_set_y',data=Y_test)
