import numpy as np
import pandas as pd
import os

# text

warm_list = np.load('./Data/www/warm_list.npy')
cold_list = np.load('./Data/www/cold_list.npy')
feat_npy = np.load('./Data/www/text_features.npy')

items = list(warm_list) + list(cold_list)

i_df = pd.DataFrame(items)
i_df.rename(columns={0:'id'}, inplace=True)

feat = pd.DataFrame(feat_npy)
feat.rename(columns={0:'id', 1:'embedding'}, inplace=True)

final = pd.merge(i_df, feat, on=['id'])
feat_final = np.array(final['embedding'])

f = np.empty([len(items), feat_npy[0][1].size])
for i, row in enumerate(feat_final):
    f[i] = row

print(f.shape)
np.save('/data/private/CLCRec/Data/www/feat_t.npy', f)

# pickle vit
import torch
import pickle
import os
filepath = '/data/private'
movielens_1 = 'ViT_movielens_5038_221006.pkl'
movielens_2 = 'ViT_movielens_2_2022-10-06_22:02:28.pkl'
yahoo = 'ViT_yahoo_752_2022-10-06_18:41:03.pkl'

with open(file=os.path.join(filepath, movielens_1), mode='rb') as f:
    data1 = pickle.load(f)
with open(file=os.path.join(filepath, movielens_2), mode='rb') as f:
    data2 = pickle.load(f)
with open(file=os.path.join(filepath, yahoo), mode='rb') as f:
    data3 = pickle.load(f)

dataset1 = data1['Video_Embedding_ViT']
dataset = {k:v.squeeze().cpu().detach().numpy() \
    for k, v in dataset1.items()}

dataset2 = data2['Video_Embedding_ViT']
dataset2 = {k:v.squeeze().cpu().detach().numpy() \
    for k, v in dataset2.items()}
dataset3 = data2['Video_Embedding_ViT']
dataset3 = {k:v.squeeze().cpu().detach().numpy() \
    for k, v in dataset3.items()}
dataset.update(dataset2)
dataset.update(dataset3)

f = np.empty([len(items), 768])
for e, i in enumerate(items):
    f[e] = dataset[i]
np.save('/data/private/CLCRec/Data/www/feat_v.npy', f)

    