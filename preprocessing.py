import json

import numpy as np
from os import listdir
from PIL import Image

paths = ("TRAIN_SET/Fedor_no_annot/Sag-CUBE-T2/", "TRAIN_SET/Jane/AX-T2-FLAIR-01-03-17/" ,
               "TRAIN_SET/Jane/AX-T2-FLAIR-24-09-18/", "TRAIN_SET/Marianne/10-09-17/AX-FSE-T2-10-09-17/",
               "TRAIN_SET/Marianne/10-09-17/AX-T2-FLAIR-10-09-17/", "TRAIN_SET/Marianne/25-12-16/AX-FSE-T2-25-12-16/",
               "TRAIN_SET/Marianne/25-12-16/AX-T1-SE+C-25-12-16/", "TRAIN_SET/Marianne/25-12-16/AX-T2-FLAIR-25-12-16/" )


filenames_train = []
filenames_target = []
for i in paths:
    filenames_train.append(sorted(listdir(i+'scans/'), key= lambda x: int(x.split('.')[-2])))
    filenames_target.append(sorted(listdir(i+'rois/'), key= lambda x: int(x.split('.')[-2])))

X_train = []
X_target = []

f = open("std_mean_dict.json", "r")


std_mean_dict = json.load(f)
print(std_mean_dict)

for k, i in enumerate(filenames_train):
    test_array = []
    for j in i:
        print(i)
        im = Image.open(paths[k] + 'scans/' + j)
        im = im.crop([193, 193, 1633, 1633])
        im = im.resize((240, 240), Image.LANCZOS)
        im = np.asarray(im.convert('L'))
        im = (im - std_mean_dict['m']) / std_mean_dict['std']
        test_array.append(im.astype(np.float32))
    name = '_'.join(paths[k].split('/')[1:])
    np.save(  f'Arrays/{name}train'  ,np.asarray(test_array))

# std_mean_dict['m'] = np.mean(X_train)
# std_mean_dict['std'] = np.std(X_train)
#
# f = open("std_mean_dict.json","w")
# f.write(json.dumps(std_mean_dict))
# f.close()
for k, i in enumerate(filenames_target):
    test_array = []
    for j in i:
        print(i)
        im = Image.open(paths[k] + 'rois/' + j)
        im = im.crop([193, 193, 1633, 1633])
        im = im.resize((240, 240), Image.LANCZOS)
        im = np.asarray(im.convert('L'))
        test_array.append(im)


    name = '_'.join(paths[k].split('/')[1:])
    np.save(  f'Arrays/{name}target'  ,np.asarray(test_array))

# X_train = np.asarray(X_train)
# X_target = np.asarray(X_target)
#
# print(X_train.shape)
# print(X_target.shape)

# m = np.mean(X_train)
# s = np.std(X_train)
#
# for e, i in enumerate(X_train):
#     img = (i - m) / s
#     img = img.astype(np.float32)
#     X_train[e] = img
#
# print(X_train[0])



