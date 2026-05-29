import argparse
import os

import colmap_read_model as read_model
import numpy as np


def pose_transform(basedir, idx):
    if len(idx) != 24:
        print('Error idx number')
        return
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    print(poses_arr.shape)
    poses_arr_trans = np.zeros_like(poses_arr)
    for i in range(poses_arr.shape[0]):
        poses_arr_trans[i] = poses_arr[idx[i] - 1]
    poses_arr_trans = np.array(poses_arr_trans)
    np.save(os.path.join(basedir, 'poses_bounds_trans.npy'), poses_arr_trans)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, 
                        help='COLMAP Directory')

    args = parser.parse_args()

    basedir = args.data_path
    pose_transform(basedir, [5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7])