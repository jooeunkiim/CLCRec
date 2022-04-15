import numpy as np
import pandas as pd
import random
import os

SEED = 42

def get_filenames(filepath, n_items = None):
    lst = os.listdir(filepath)
    fnames = []
    movie_ids = {}
    for movie in lst:
        if n_items:
            if len(movie_ids) > n_items: break
        files = os.listdir(os.path.join(filepath, movie))
        jpgs = [f for f in files if f.endswith(".jpg")]
        if len(jpgs) > 0:
            movie_ids[int(movie)] = jpgs
            fnames.extend(jpgs)
    return movie_ids, fnames


def filter_ratings(raw_ratings, raw_movies):
    num_movies = 5000
    raw_ratings = pd.read_csv(raw_ratings)
    data = raw_ratings[(raw_ratings['movielens_id'].isin(raw_movies)) & (raw_ratings['rating'] >= 3.5)]
    usr_grp = data.groupby('userId')
    # itm_grp = data.groupby('movielens_id')
    movies = list(set(data['movielens_id']))
    if len(movies) > num_movies:
        random.seed(SEED)
        movies = random.sample(movies, num_movies)
    movie_ids = {item:1 for item in movies}
    data = data[data['movielens_id'].apply(lambda x: x in movie_ids)]
    if len(movies) == num_movies:
        active = usr_grp.size()[usr_grp.size() >= 50]
        data = data[data['userId'].apply(lambda x: x in active)]
        print(len(active), "users and", len(set(data['movielens_id'])), "items left")
    movies = list(set(data['movielens_id']))
    return data, movies


def warm_cold_split(data, warm_items, cold_items):
    cold_ids = {item:1 for item in cold_items}
    cold_idx = data['movielens_id'].apply(lambda x: x in cold_ids)
    cold_data = data[cold_idx]
    warm_data = data[~cold_idx]

    n_cold = len(cold_data)
    cold_val = cold_data[:int(n_cold // 2)]
    cold_test = cold_data[int(n_cold // 2):]

    print("warm items:", len(warm_items), ", cold items:", len(cold_items))

    random.seed(SEED)
    warm_data = warm_data.sample(frac=1).reset_index(drop=True)
    n_warm = len(warm_data)
    warm_train = warm_data[:int(n_warm*0.8)]
    warm_val = warm_data[int(n_warm*0.8):int(n_warm*0.9)]
    warm_test = warm_data[int(n_warm*0.9):]

    print("data points - warm train:", len(warm_train), ", warm val:", len(warm_val), ", warm test:", len(warm_test),\
        ", cold val:", len(cold_val), ", cold test:", len(cold_test))
    return warm_train, warm_val, warm_test, cold_val, cold_test

def data2npy(data, movies, data_dir):
    users = list(set(data['userId']))
    useridx = dict(zip(users, np.arange(len(users))))

    warm_items = movies[int(len(movies) * 0.15):]
    cold_items = movies[:int(len(movies) * 0.15)]
    warm_set = np.arange(len(users), len(users)+len(warm_items)) ## warm_set.npy
    cold_set = np.arange(len(users)+len(warm_items), len(users)+len(warm_items)+len(cold_items)) ## cold_set.npy
    warmidx = dict(zip(warm_items, warm_set))
    coldidx = dict(zip(cold_items, cold_set))

    warm_train, warm_val, warm_test, cold_val, cold_test = warm_cold_split(data, warm_items, cold_items)
    
    train = np.array(warm_train[['userId', 'movielens_id']])
    train = np.array(list(map(lambda x: [useridx[x[0]], warmidx[x[1]]], train))) ## train.npy

    ui_dict = {u:[] for u in useridx.values()}
    for row in np.array(warm_train[['userId', 'movielens_id']]):
        ui_dict[useridx[row[0]]].append(warmidx[row[1]])
    
    ui_train = {u:set(i) for u,i in ui_dict.items()}
    val_full = [[u] for u in useridx.values() ]## val_full.npy
    val_warm = [[u] for u in useridx.values()] ## val_warm.npy
    val_cold = [[u] for u in useridx.values()] ## val_cold.npy
    for row in np.array(warm_val[['userId', 'movielens_id']]):
        ui_dict[useridx[row[0]]].append(warmidx[row[1]])
        val_full[useridx[row[0]]].append(warmidx[row[1]])
        val_warm[useridx[row[0]]].append(warmidx[row[1]])
    for row in np.array(cold_val[['userId', 'movielens_id']]):
        ui_dict[useridx[row[0]]].append(coldidx[row[1]])
        val_full[useridx[row[0]]].append(coldidx[row[1]])
        val_cold[useridx[row[0]]].append(coldidx[row[1]])

    test_full = [[u] for u in useridx.values() ]## test_full.npy
    test_warm = [[u] for u in useridx.values()] ## test_warm.npy
    test_cold = [[u] for u in useridx.values()] ## test_cold.npy
    for row in np.array(warm_test[['userId', 'movielens_id']]):
        ui_dict[useridx[row[0]]].append(warmidx[row[1]])
        test_full[useridx[row[0]]].append(warmidx[row[1]])
        test_warm[useridx[row[0]]].append(warmidx[row[1]])
    for row in np.array(cold_test[['userId', 'movielens_id']]):
        ui_dict[useridx[row[0]]].append(coldidx[row[1]])
        test_full[useridx[row[0]]].append(coldidx[row[1]])
        test_cold[useridx[row[0]]].append(coldidx[row[1]])
    
    np.save(data_dir+"item_list.npy", movies)
    np.save(data_dir+"warm_set.npy", warm_set)
    np.save(data_dir+"cold_set.npy", cold_set)
    np.save(data_dir+"train.npy", train)
    np.save(data_dir+"user_item_train_dict.npy", ui_train)
    np.save(data_dir+"user_item_all_dict.npy", ui_dict)
    np.save(data_dir+"val_full.npy", val_full)
    np.save(data_dir+"val_warm.npy", val_warm)
    np.save(data_dir+"val_cold.npy", val_cold)
    np.save(data_dir+"test_full.npy", test_full)
    np.save(data_dir+"test_warm.npy", test_warm)
    np.save(data_dir+"test_cold.npy", test_cold)


if __name__=='__main__':
    # raw_movies, contents = get_filenames('/data/dataset/recsys/videos')
    # data, movies = filter_ratings("/data/dataset/recsys/ratings_tv.csv", raw_movies)
    # data2npy(data, movies, "/data/projects/CLCRec/Data/movie/")
    
    raw_movies, contents = get_filenames('/data/private/videos')
    data, movies = filter_ratings("/data/private/ratings_video.csv", raw_movies)
    data2npy(data, movies, "/data/private/CLCRec/Data/movie/")

