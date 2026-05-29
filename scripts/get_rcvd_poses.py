import argparse
import os
import imageio
import numpy as np
import math



def read_rcvd_poses(basedir):
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    h, w = sh[0], sh[1]

    pose0 = [os.path.join(basedir, 'camera_params', f) for f in sorted(os.listdir(os.path.join(basedir, 'camera_params'))) \
            if f.endswith('npz')][0]
    pose_params0 = np.load(pose0)
    vFOV = pose_params0['vF']
    f = h / (2. * math.tan(vFOV / 2.))
    hwf = np.array([h, w, f]).reshape([3,1])
    
    w2c_mats = []
    bounds_mats = []
    posedirs = [os.path.join(basedir, 'camera_params', f) for f in sorted(os.listdir(os.path.join(basedir, 'camera_params'))) \
                if f.endswith('npz')]
    for i in range(len(posedirs)):
        pose_params = np.load(posedirs[i])
        m = pose_params['world2cam']
        w2c_mats.append(m)

        bounds = np.array([40., 120.])
        bounds_mats.append(bounds)

    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], 
                                        [1,1,poses.shape[-1]])], 1)
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], 
                            -poses[:, 2:3, :], 
                            poses[:, 3:4, :], 
                            poses[:, 4:5, :]], 1)
    
    save_arr = []
    for i in range((poses.shape[2])):
        save_arr.append(np.concatenate([poses[..., i].ravel(), bounds_mats[i]], 0))
    
    save_arr = np.array(save_arr)
    print(save_arr.shape)
    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)



if __name__=='__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_path", type=str, 
    #                     help='COLMAP Directory')

    # args = parser.parse_args()

    basedir = '/data/sunhuiqiang/nsff_learn_pose/data/0/dense' #args.data_path 
    read_rcvd_poses(basedir)
    print( 'Done with rcvd2poses' )