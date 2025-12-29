import numpy as np
import torch
from math import radians, cos, sin, asin, sqrt
import joblib
from torch.nn.utils.rnn import pad_sequence

max_len = 100  # max traj len; i.e., M


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


def euclidean(point, each):
    lon1, lat1, lon2, lat2 = point[2], point[1], each[2], each[1]
    return np.sqrt((lon1 - lon2)**2 + (lat1 - lat2)**2)


def rst_mat1(traj, poi):
    # traj (*M, [u, l, t]), poi(L, [l, lat, lon])
    mat = np.zeros((len(traj), len(traj), 2))
    for i, item in enumerate(traj):
        for j, term in enumerate(traj):
            poi_item, poi_term = poi[item[1] - 1], poi[term[1] - 1]  # retrieve poi by loc_id
            mat[i, j, 0] = haversine(lon1=poi_item[2], lat1=poi_item[1], lon2=poi_term[2], lat2=poi_term[1])
            mat[i, j, 1] = abs(item[2] - term[2])
    return mat  # (*M, *M, [dis, tim])


def rs_mat2s(poi, l_max):
    # poi(L, [l, lat, lon])
    candidate_loc = np.linspace(1, l_max, l_max)  # (L)
    mat = np.zeros((l_max, l_max))  # mat (L, L)
    for i, loc1 in enumerate(candidate_loc):
        print(i) if i % 100 == 0 else None
        for j, loc2 in enumerate(candidate_loc):
            poi1, poi2 = poi[int(loc1) - 1], poi[int(loc2) - 1]  # retrieve poi by loc_id
            mat[i, j] = haversine(lon1=poi1[2], lat1=poi1[1], lon2=poi2[2], lat2=poi2[1])
    return mat  # (L, L)


def rt_mat2t(traj_time):  # traj_time (*M+1) triangle matrix
    # construct a list of relative times w.r.t. causality
    mat = np.zeros((len(traj_time)-1, len(traj_time)-1))
    for i, item in enumerate(traj_time):  # label
        if i == 0:
            continue
        for j, term in enumerate(traj_time[:i]):  # data
            mat[i-1, j] = np.abs(item - term)
    return mat  # (*M, *M)


import argparse
import sys

def process_traj(dataset_path, dname):  # modified signature
    # dataset_path: directory containing the .npy files (e.g., data/TSMC_NYC)
    # dname: name of the dataset (e.g., NYC)
    
    data_file = os.path.join(dataset_path, dname + '.npy')
    poi_file = os.path.join(dataset_path, dname + '_POI.npy')
    
    print(f"Loading data from {data_file}")
    data = np.load(data_file)
    
    # add the code below if you are using dividing time into minutes instead of hours
    data[:, -1] = np.array(data[:, -1]/60, dtype=int)
    
    poi = np.load(poi_file)
    num_user = data[:, 0].max() # Use max of col 0, faster than sorting whole array? original was data[-1, 0] assuming sorted
    
    # Original code assumed sorted by user?
    # "data_user = data[:, 0]"
    # "u_max... = np.max(data[:, 0])"
    
    # Let's trust np.max for safety
    u_max = np.max(data[:, 0])
    l_max = np.max(data[:, 1])
    num_user = u_max

    data_user = data[:, 0]  # user_id sequence in data
    trajs, labels, mat1, mat2t, lens = [], [], [], [], []

    # Original loop iterated range(num_user+1). 
    # If user IDs are not contiguous in filtered data (e.g. 1, 3, 5), this loop will find empty user_traj for 2, 4.
    # preprocess_raw.py maps to 1..N contiguous. So this is fine.
    
    for u_id in range(1, int(num_user)+1):
        # ... (rest of loop logic is same)
        init_mat1 = np.zeros((max_len, max_len, 2))
        init_mat2t = np.zeros((max_len, max_len))
        user_traj = data[np.where(data_user == u_id)]
        
        if len(user_traj) == 0: continue

        user_traj = user_traj[np.argsort(user_traj[:, 2])].copy()
        
        print(u_id, len(user_traj)) if u_id % 100 == 0 else None

        if len(user_traj) > max_len + 1:
            user_traj = user_traj[-max_len-1:]

        user_len = len(user_traj[:-1])
        user_mat1 = rst_mat1(user_traj[:-1], poi)
        user_mat2t = rt_mat2t(user_traj[:, 2])
        init_mat1[0:user_len, 0:user_len] = user_mat1
        init_mat2t[0:user_len, 0:user_len] = user_mat2t

        trajs.append(torch.LongTensor(user_traj)[:-1])
        mat1.append(init_mat1)
        mat2t.append(init_mat2t)
        labels.append(torch.LongTensor(user_traj[1:, 1]))
        lens.append(user_len-2)

    poi_coords = poi[:, 1:3]

    zipped = zip(*sorted(zip(trajs, mat1, mat2t, labels, lens), key=lambda x: len(x[0]), reverse=True))
    trajs, mat1, mat2t, labels, lens = zipped
    trajs, mat1, mat2t, labels, lens = list(trajs), list(mat1), list(mat2t), list(labels), list(lens)
    trajs = pad_sequence(trajs, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)

    data_packet = [trajs, np.array(mat1), poi_coords, np.array(mat2t), labels, np.array(lens), u_max, l_max]
    
    data_pkl = os.path.join(dataset_path, dname + '_data.pkl')
    print(f"Saving processed data to {data_pkl}")
    # open(data_pkl, 'a') # Original had this weird touch
    # Just write directly
    with open(data_pkl, 'wb') as pkl:
        joblib.dump(data_packet, pkl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help='Path to dataset directory (e.g. data/TSMC_NYC)')
    parser.add_argument('dataset_name', type=str, help='Dataset name (e.g. NYC)')
    args = parser.parse_args()
    
    import os # ensure os is imported if not already in global scope or just relies on top level
    # Top level usually has import numpy, torch, etc. Need os.
    
    process_traj(args.dataset_path, args.dataset_name)

