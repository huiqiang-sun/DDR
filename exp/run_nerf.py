import os, sys
import numpy as np
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import cv2
from kornia import create_meshgrid

from render_utils import *
from run_nerf_helpers import *
from load_llff import *
from sampling_argmax import *
from poses import *
from focal import *

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)
DEBUG = False

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',

                        help='input data directory')
    parser.add_argument("--render_lockcam_slowmo", action='store_true', 
                        help='render fixed view + slowmo')
    parser.add_argument("--render_slowmo_bt", action='store_true', 
                        help='render space-time interpolation')

    parser.add_argument("--final_height", type=int, default=288, 
                        help='training image height, default is 512x288')
    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=300, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*16, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*16, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_bt", action='store_true', 
                        help='render bullet time')

    parser.add_argument("--render_test", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    parser.add_argument("--target_idx", type=int, default=10, 
                        help='target_idx')
    parser.add_argument("--num_extra_sample", type=int, default=512, 
                        help='num_extra_sample')
    parser.add_argument("--decay_depth_w", action='store_true', 
                        help='decay depth weights')
    parser.add_argument("--use_motion_mask", action='store_true', 
                        help='use motion segmentation mask for hard-mining data-driven initialization')
    parser.add_argument("--decay_optical_flow_w", action='store_true', 
                        help='decay optical flow weights')

    parser.add_argument("--w_depth",   type=float, default=0.04, 
                        help='weights of depth loss')
    parser.add_argument("--w_optical_flow", type=float, default=0.02, 
                        help='weights of optical flow loss')
    parser.add_argument("--w_sm", type=float, default=0.1, 
                        help='weights of scene flow smoothness')
    parser.add_argument("--w_sf_reg", type=float, default=0.1, 
                        help='weights of scene flow regularization')
    parser.add_argument("--w_cycle", type=float, default=0.1, 
                        help='weights of cycle consistency')
    parser.add_argument("--w_prob_reg", type=float, default=0.1, 
                        help='weights of disocculusion weights')

    parser.add_argument("--w_entropy", type=float, default=1e-3, 
                        help='w_entropy regularization weight')

    parser.add_argument("--decay_iteration", type=int, default=50, 
                        help='data driven priors decay iteration * 1000')

    parser.add_argument("--chain_sf", action='store_true', 
                        help='5 frame consistency if true, \
                             otherwise 3 frame consistency')

    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=50)

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=1000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')

    parser.add_argument("--w_arg", type=float, default=0.01, help='w_arg regularization weight')
    parser.add_argument("--decay_arg_w", action='store_true', help='decay arg weights')
    parser.add_argument("--num_samples", type=int, default=30, help='sample number in gumbel softmax')
    parser.add_argument("--basis_type", type=str, default='tri', help='sub-distribution type')
    parser.add_argument("--tau", type=float, default=1., help='temperature')
    parser.add_argument("--decay_tau", action='store_true', help='temperature decay')


    
    parser.add_argument('--learn_poses', type=bool, default=False, help='learn poses?')
    parser.add_argument('--learn_focal', type=bool, default=False, help='learn focal?')
    parser.add_argument('--pose_lr', default=0.001, type=float)
    parser.add_argument('--pose_milestones', default=list(range(0, 10000, 100)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--pose_lr_gamma', type=float, default=0.9, help="learning rate milestones gamma")
    parser.add_argument('--focal_lr', default=0.001, type=float)
    parser.add_argument('--focal_milestones', default=list(range(0, 10000, 100)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--focal_lr_gamma', type=float, default=0.9, help="learning rate milestones gamma")
    parser.add_argument("--w_gradient", type=float, default=0.01, help='w_gradient regularization weight')
    parser.add_argument("--decay_gradient_w", action='store_true', help='decay gradient weights')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()
    print(args.learn_focal)

    # Load data
    if args.dataset_type == 'llff':
        target_idx = args.target_idx
        #datadir = '/data1/sunhuiqiang/Neural-Scene-Flow-Fields-main/data/kid-running/dense'
        images, depths, masks, poses, bds, render_poses, ref_c2w, motion_coords = load_llff_data(
                                                            args.datadir, 
                                                            args.start_frame, args.end_frame,
                                                            args.factor,
                                                            target_idx=target_idx,
                                                            recenter=True, bd_factor=.9,
                                                            spherify=args.spherify, 
                                                            final_height=args.final_height)

        # idx = [5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7]
        # poses_temp = np.zeros_like(poses)
        # for i in range(poses.shape[0]):
        #     poses_temp[i] = poses[idx[i] - 1]
        # poses = poses_temp

        c2w = poses[target_idx, :, :]
        render_poses = render_wander_path(c2w)
        render_poses = np.array(render_poses).astype(np.float32)
        
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        i_test = []
        i_val = [] #i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.percentile(bds[:, 0], 5) * 0.8 #np.ndarray.min(bds) #* .9
            far = np.percentile(bds[:, 1], 95) * 1.1 #np.ndarray.max(bds) #* 1.
        else:
            near = 0.
            far = 1.

        print('NEAR FAR', near, far)
    else:
        print('ONLY SUPPORT LLFF!!!!!!!!')
        sys.exit()


    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir and copy the config file
    basedir = args.basedir
    args.expname = args.expname + '_F%02d-%02d'%(args.start_frame, args.end_frame)
    
    # args.expname = args.expname + '_sigma_rgb-%.2f'%(args.sigma_rgb) \
                # + '_use-rgb-w_' + str(args.use_rgb_w) + '_F%02d-%02d'%(args.start_frame, args.end_frame)
    
    expname = args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    
    if args.learn_poses:
        pose_history_dir = os.path.join(basedir, expname, 'pose_history')
        os.makedirs(pose_history_dir, exist_ok=True)

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }

    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    if args.learn_poses:
        num_img = poses.shape[0]
        pose_param_net = LearnPose(num_img, torch.Tensor(poses)).to(device)
        device_ids = list(range(torch.cuda.device_count()))
        pose_param_net = torch.nn.DataParallel(pose_param_net, device_ids=device_ids)
        optimizer_pose = torch.optim.Adam(pose_param_net.parameters(), lr=args.pose_lr)

        if args.ft_path is not None and args.ft_path!='None':
            ckpts = [args.ft_path]
        else:
            ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]

            print('Reloading poses params from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            pose_param_net.load_state_dict(ckpt['poses_net_state_dict'])
            optimizer_pose.load_state_dict(ckpt['poses_optimizer_state_dict'])
        
        scheduler_pose = torch.optim.lr_scheduler.MultiStepLR(optimizer_pose, milestones=args.pose_milestones,
                                                              gamma=args.pose_lr_gamma)

        
    if args.learn_focal:
        focal_net = LearnFocal(init_focal=torch.tensor(focal))
        device_ids = list(range(torch.cuda.device_count()))
        focal_net = torch.nn.DataParallel(focal_net, device_ids=device_ids)
        optimizer_focal = torch.optim.Adam(focal_net.parameters(), lr=args.focal_lr)

        if args.ft_path is not None and args.ft_path!='None':
            ckpts = [args.ft_path]
        else:
            ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]

            print('Reloading poses params from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            focal_net.load_state_dict(ckpt['focal_net_state_dict'])
            optimizer_focal.load_state_dict(ckpt['focal_optimizer_state_dict'])
        
        scheduler_focal = torch.optim.lr_scheduler.MultiStepLR(optimizer_focal, milestones=args.focal_milestones,
                                                               gamma=args.focal_lr_gamma)


    # 渲染时间固定，视角插值的新图像
    if args.render_bt:
        print('RENDER VIEW INTERPOLATION')            
        print('target_idx ', target_idx)

        num_img = float(poses.shape[0])
        img_idx_embed = target_idx/float(num_img) * 2. - 1.0    #将时间归一化到[-1,1]范围

        testsavedir = os.path.join(basedir, expname, 
                                'render-spiral-frame-%03d'%\
                                target_idx + '_{}_{:06d}'.format('test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            hwf_temp = ref_c2w[:,4:5]
            if args.learn_focal:
                focal = focal_net()
                hwf = [H, W, focal]
                hwf_temp = torch.tensor(hwf).reshape([3,1])
                hwf_temp = hwf_temp.cpu().numpy()
                print(hwf_temp)
            if args.learn_poses:
                c2w = pose_param_net(target_idx).cpu().numpy()  
                render_poses = render_wander_path(c2w, hwf=hwf_temp)
                render_poses = np.array(render_poses).astype(np.float32)
                render_poses = torch.Tensor(render_poses).to(device)
            print(c2w.shape, c2w)
            render_bullet_time(render_poses, img_idx_embed, num_img, hwf, 
                               args.chunk, render_kwargs_test, 
                               gt_imgs=images, savedir=testsavedir, 
                               render_factor=args.render_factor)

        return

    # 渲染视角固定，时间插值的新图像
    if args.render_lockcam_slowmo:
        print('RENDER TIME INTERPOLATION')

        num_img = float(poses.shape[0])
        print('target_idx ', target_idx)

        testsavedir = os.path.join(basedir, expname, 'render-lockcam-slowmo')
        os.makedirs(testsavedir, exist_ok=True)
        with torch.no_grad():
            if args.learn_focal:
                focal = focal_net()
                hwf = [H, W, focal]
            if args.learn_poses:
                c2w = pose_param_net(target_idx)
                ref_c2w = torch.Tensor(c2w).to(device)
            
            render_lockcam_slowmo(ref_c2w, num_img, hwf, 
                                  args.chunk, render_kwargs_test, 
                                  gt_imgs=images, savedir=testsavedir, 
                                  render_factor=args.render_factor,
                                  target_idx=target_idx)

            return 

    # 渲染视角和时间同时插值的图像
    if args.render_slowmo_bt:
        print('RENDER SLOW MOTION')

        curr_ts = 0
        # render_poses大小为[N,3,5]表示输入图像的外参；
        # bt_poses大小为[400,4,4]，表示将一个输入图像的外参旋转到周围40个不同的位置，用于渲染再复制10份
        render_poses = poses #torch.Tensor(poses).to(device)

        with torch.no_grad():

            testsavedir = os.path.join(basedir, expname, 
                                    'render-slowmo_bt_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            images = torch.Tensor(images)#.to(device)
            hwf_temp = ref_c2w[:,4:5]
            if args.learn_focal:
                focal = focal_net()
                hwf = [H, W, focal]
                hwf_temp = torch.tensor(hwf).reshape([3,1])
                hwf_temp = hwf_temp.cpu().numpy()
            bt_poses = create_bt_poses(hwf_temp) 
            bt_poses = bt_poses * 10

            print('render poses shape', render_poses.shape)
            render_slowmo_bt(depths, render_poses, bt_poses, pose_param_net, 
                            hwf, args.chunk, render_kwargs_test,
                            gt_imgs=images, savedir=testsavedir, 
                            render_factor=args.render_factor, 
                            target_idx=10)
            # print('Done rendering', i,testsavedir)

        return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    # Move training data to GPU
    images = torch.Tensor(images)#.to(device)
    depths = torch.Tensor(depths)#.to(device)
    masks = 1.0 - torch.Tensor(masks).to(device)

    poses = torch.Tensor(poses).to(device)

    N_iters = 300 * 1000 + 1 #1000000
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    # uv_grid大小为[H, W, 2]，存放图像坐标，最后一维的两个元素为坐标[W,H]
    uv_grid = create_meshgrid(H, W, normalized_coordinates=False)[0].cuda() # (H, W, 2)

    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    num_img = float(images.shape[0])
    
    decay_iteration = max(args.decay_iteration, 
                          args.end_frame - args.start_frame)
    decay_iteration = min(decay_iteration, 250)

    chain_bwd = 0

    for i in range(start, N_iters):
        chain_bwd = 1 - chain_bwd
        time0 = time.time()
        print('expname ', expname, ' chain_bwd ', chain_bwd, ' focal ', focal, 
             ' lindisp ', args.lindisp, ' decay_iteration ', decay_iteration)
        print('Random FROM SINGLE IMAGE')
        # Random from one image
        img_i = np.random.choice(i_train)

        if i % (decay_iteration * 1000) == 0:
            torch.cuda.empty_cache()

        # target表示抽取到的图像，大小为[W,H,C]；pose表示该图片的相机外参，大小为[3,4]
        # depth_gt表示该图像对应的深度图，大小为[W,H]；hard_coords大小为[M,2]；mask_gt大小为[W,H]
        target = images[img_i].cuda()
        # pose = poses[img_i, :3,:4]
        if args.learn_poses:
            pose = pose_param_net(img_i)
        else:
            pose = poses[img_i, :3,:4]
        if args.learn_focal:
            focal = focal_net()
            hwf = [H, W, focal]
        depth_gt = depths[img_i].cuda()
        hard_coords = torch.Tensor(motion_coords[img_i]).cuda()
        mask_gt = masks[img_i].cuda()

        # 得到图像对应的前向与后向flow以及mask的gt值，对于第一张和最后一张图片只有半边
        if img_i == 0:
            flow_fwd, fwd_mask = read_optical_flow(args.datadir, img_i, 
                                                args.start_frame, fwd=True)
            flow_bwd, bwd_mask = np.zeros_like(flow_fwd), np.zeros_like(fwd_mask)
        elif img_i == num_img - 1:
            flow_bwd, bwd_mask = read_optical_flow(args.datadir, img_i, 
                                                args.start_frame, fwd=False)
            flow_fwd, fwd_mask = np.zeros_like(flow_bwd), np.zeros_like(bwd_mask)
        else:
            flow_fwd, fwd_mask = read_optical_flow(args.datadir, 
                                                img_i, args.start_frame, 
                                                fwd=True)
            flow_bwd, bwd_mask = read_optical_flow(args.datadir, 
                                                img_i, args.start_frame, 
                                                fwd=False)

        # # ======================== TEST 
        TEST = False
        if TEST:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            print('CHECK DEPTH and FLOW and exiting')
            print(images[img_i].shape)
            print(flow_fwd.shape, img_i)

            warped_im2 = warp_flow(images[img_i + 1].cpu().numpy(), flow_fwd)
            warped_im0 = warp_flow(images[img_i - 1].cpu().numpy(), flow_bwd)
            mask_gt = masks[img_i].cpu().numpy()

            plt.figure(figsize=(12, 6))

            plt.subplot(2, 3, 1)
            plt.imshow(target.cpu().numpy())
            plt.subplot(2, 3, 4)
            plt.imshow(depth_gt.cpu().numpy(), cmap='jet') 

            plt.subplot(2, 3, 2)
            plt.imshow(flow_to_image(flow_fwd)/255. * fwd_mask[..., np.newaxis])

            plt.subplot(2, 3, 3)
            plt.imshow(flow_to_image(flow_bwd)/255. * bwd_mask[..., np.newaxis])

            plt.subplot(2, 3, 5)
            plt.imshow(mask_gt, cmap='gray')

            cv2.imwrite('im_%d.jpg'%(img_i),
                        np.uint8(np.clip(target.cpu().numpy()[:, :, ::-1], 0, 1) * 255))
            cv2.imwrite('im_%d_warp.jpg'%(img_i + 1), 
                        np.uint8(np.clip(warped_im2[:, :, ::-1], 0, 1) * 255))
            cv2.imwrite('im_%d_warp.jpg'%(img_i - 1), 
                        np.uint8(np.clip(warped_im0[:, :, ::-1], 0, 1) * 255))
            plt.savefig('depth_flow_%d.jpg'%img_i)
            sys.exit()

        #  END OF TEST
        flow_fwd = torch.Tensor(flow_fwd).cuda()
        fwd_mask = torch.Tensor(fwd_mask).cuda()
    
        flow_bwd = torch.Tensor(flow_bwd).cuda()
        bwd_mask = torch.Tensor(bwd_mask).cuda()
        # more correct way for flow loss
        # flow_fwd和flow_bwd表示当前图像的点在下一帧和上一帧图像的哪个地方（真实值）
        flow_fwd = flow_fwd + uv_grid
        flow_bwd = flow_bwd + uv_grid

        if N_rand is not None:
            rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)
            
            # coords大小为[H*W, 2]，顺序存放图像每个点的图像坐标[H,W]
            # 顺序为从左到右，从上到下
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)

            if args.use_motion_mask and i < decay_iteration * 1000:
                print('HARD MINING STAGE !')
                num_extra_sample = args.num_extra_sample
                print('num_extra_sample ', num_extra_sample)
                select_inds_hard = np.random.choice(hard_coords.shape[0], 
                                                    size=[min(hard_coords.shape[0], 
                                                        num_extra_sample)], 
                                                    replace=False)  # (N_rand,)
                select_inds_all = np.random.choice(coords.shape[0], 
                                                size=[N_rand], 
                                                replace=False)  # (N_rand,)

                select_coords_hard = hard_coords[select_inds_hard].long()
                select_coords_all = coords[select_inds_all].long()

                # select_coords大小为[N_rand+num_extra_sample, 2]，表示随机抽取的图像坐标点
                # 其中对于hard_coords里面的点有多一次被抽取到的可能
                # 之后将N_rand+num_extra_sample简称为N_rand
                select_coords = torch.cat([select_coords_all, select_coords_hard], 0)

            else:
                select_inds = np.random.choice(coords.shape[0], 
                                            size=[N_rand], 
                                            replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
            
            # rays_o和rays_d大小为[N_rand,3]，表示被抽取到的那些点的o和d值
            # batch_rays大小为[2,N_rand,3]，将rays_o和rays_d拼接起来
            # target_rgb大小为[N_rand,3]，代表抽取到的那些点的rgb值；
            # target_depth大小为[N_rand]，表示抽取到的那些点的深度值
            # target_of_fwd、target_fwd_mask、target_of_bwd、target_bwd_mask为抽取到的点的
            # 前向和后向场景流以及mask，大小都为[N_rand,...]
            rays_o = rays_o[select_coords[:, 0], 
                            select_coords[:, 1]]  
            rays_d = rays_d[select_coords[:, 0], 
                            select_coords[:, 1]]  
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_rgb = target[select_coords[:, 0], 
                                select_coords[:, 1]]
            target_depth = depth_gt[select_coords[:, 0], 
                                select_coords[:, 1]]
            target_mask = mask_gt[select_coords[:, 0], 
                                select_coords[:, 1]].unsqueeze(-1)

            target_of_fwd = flow_fwd[select_coords[:, 0], 
                                     select_coords[:, 1]]
            target_fwd_mask = fwd_mask[select_coords[:, 0], 
                                     select_coords[:, 1]].unsqueeze(-1)#.repeat(1, 2)

            target_of_bwd = flow_bwd[select_coords[:, 0], 
                                     select_coords[:, 1]]
            target_bwd_mask = bwd_mask[select_coords[:, 0], 
                                     select_coords[:, 1]].unsqueeze(-1)#.repeat(1, 2)

        # 得到这一帧图像的序号（归一化到[-1,1]之间）
        img_idx_embed = img_i/num_img * 2. - 1.0

        #####  Core optimization loop  #####
        if args.chain_sf and i > decay_iteration * 1000 * 2:
            chain_5frames = True
        else:
            chain_5frames = False

        print('chain_5frames ', chain_5frames, ' chain_bwd ', chain_bwd)

        ret = render(img_idx_embed, 
                     chain_bwd, 
                     chain_5frames,
                     num_img, H, W, focal, 
                     chunk=args.chunk, 
                     rays=batch_rays,
                     verbose=i < 10, retraw=True,
                     **render_kwargs_train)

        # 得到当前图像前后两帧图像的相机外参
        if args.learn_poses:
            pose_post = pose_param_net(min(img_i + 1, int(num_img) - 1))
            pose_prev = pose_param_net(max(img_i - 1, 0))
        else:
            pose_post = poses[min(img_i + 1, int(num_img) - 1), :3,:4]
            pose_prev = poses[max(img_i - 1, 0), :3,:4]

        # 得到当前图像动态区域的点在下一帧和上一帧图像的哪个地方
        render_of_fwd, render_of_bwd = compute_optical_flow(pose_post, 
                                                            pose, pose_prev, 
                                                            H, W, focal, 
                                                            ret)

        optimizer.zero_grad()
        if args.learn_poses:
            optimizer_pose.zero_grad()
        if args.learn_focal:
            optimizer_focal.zero_grad()

        # weight_map_post和weight_map_prev分别表示后一帧和前一帧的图像遮挡权重
        # weight_post和weight_prev分别表示后一帧和前一帧的空间点遮挡权重，优化时希望接近1
        weight_map_post = ret['prob_map_post']
        weight_map_prev = ret['prob_map_prev']

        weight_post = 1. - ret['raw_prob_ref2post']
        weight_prev = 1. - ret['raw_prob_ref2prev']
        # 该损失用来约束遮挡权重w，对应原论文L_pho的第二项
        prob_reg_loss = args.w_prob_reg * (torch.mean(torch.abs(ret['raw_prob_ref2prev'])) \
                                + torch.mean(torch.abs(ret['raw_prob_ref2post'])))

        # dynamic rendering loss
        # 该损失为动态渲染损失，对应原论文中L_pho的第一项
        if i <= decay_iteration * 1000:
            render_loss = img2mse(ret['rgb_map_ref_dy'], target_rgb)
            render_loss += compute_mse(ret['rgb_map_post_dy'], 
                                       target_rgb, 
                                       weight_map_post.unsqueeze(-1))
            render_loss += compute_mse(ret['rgb_map_prev_dy'], 
                                       target_rgb, 
                                       weight_map_prev.unsqueeze(-1))
        else:
            print('only compute dynamic render loss in masked region')
            weights_map_dd = ret['weights_map_dd'].unsqueeze(-1).detach()

            # dynamic rendering loss
            render_loss = compute_mse(ret['rgb_map_ref_dy'], 
                                      target_rgb, 
                                      weights_map_dd)
            render_loss += compute_mse(ret['rgb_map_post_dy'], 
                                       target_rgb, 
                                       weight_map_post.unsqueeze(-1) * weights_map_dd)
            render_loss += compute_mse(ret['rgb_map_prev_dy'], 
                                       target_rgb, 
                                       weight_map_prev.unsqueeze(-1) * weights_map_dd)

        # union rendering loss
        # 该损失为混合渲染损失，对应原论文的L_cb
        render_loss += img2mse(ret['rgb_map_ref'][:N_rand, ...], 
                               target_rgb[:N_rand, ...])

        render_loss += compute_mse(ret['rgb_map_rig'], target_rgb, target_mask)

        # 该损失为场景流循环损失，对应原论文的L_cyc
        sf_cycle_loss = args.w_cycle * compute_mae(ret['raw_sf_ref2post'], 
                                                   -ret['raw_sf_post2ref'], 
                                                   weight_post.unsqueeze(-1), dim=3)
        sf_cycle_loss += args.w_cycle * compute_mae(ret['raw_sf_ref2prev'], 
                                                    -ret['raw_sf_prev2ref'], 
                                                    weight_prev.unsqueeze(-1), dim=3)
        
        # regularization loss
        render_sf_ref2prev = torch.sum(ret['weights_ref_dy'].unsqueeze(-1) * ret['raw_sf_ref2prev'], -1)
        render_sf_ref2post = torch.sum(ret['weights_ref_dy'].unsqueeze(-1) * ret['raw_sf_ref2post'], -1)

        # 该损失用于约束场景流大小，对应原论文L_reg里面的L_min
        sf_reg_loss = args.w_sf_reg * (torch.mean(torch.abs(render_sf_ref2prev)) \
                                    + torch.mean(torch.abs(render_sf_ref2post))) 

        divsor = i // (decay_iteration * 1000)

        decay_rate = 10

        if args.decay_depth_w:
            w_depth = args.w_depth/(decay_rate ** divsor)
        else:
            w_depth = args.w_depth

        if args.decay_optical_flow_w:
            w_of = args.w_optical_flow/(decay_rate ** divsor)
        else:
            w_of = args.w_optical_flow

        # 该损失用于约束深度值，对应原论文L_data里面的L_z项
        depth_loss = w_depth * compute_depth_loss(ret['depth_map_ref_dy'], -target_depth)

        print('w_depth ', w_depth, 'w_of ', w_of)

        # 该损失为几何约束，对应原论文L_data里面的L_geo项
        if img_i == 0:
            print('only fwd flow')
            flow_loss = w_of * compute_mae(render_of_fwd, 
                                        target_of_fwd, 
                                        target_fwd_mask)#torch.sum(torch.abs(render_of_fwd - target_of_fwd) * target_fwd_mask)/(torch.sum(target_fwd_mask) + 1e-8)
        elif img_i == num_img - 1:
            print('only bwd flow')
            flow_loss = w_of * compute_mae(render_of_bwd, 
                                        target_of_bwd, 
                                        target_bwd_mask)#torch.sum(torch.abs(render_of_bwd - target_of_bwd) * target_bwd_mask)/(torch.sum(target_bwd_mask) + 1e-8)
        else:
            flow_loss = w_of * compute_mae(render_of_fwd, 
                                        target_of_fwd, 
                                        target_fwd_mask)#torch.sum(torch.abs(render_of_fwd - target_of_fwd) * target_fwd_mask)/(torch.sum(target_fwd_mask) + 1e-8)
            flow_loss += w_of * compute_mae(render_of_bwd, 
                                        target_of_bwd, 
                                        target_bwd_mask)#torch.sum(torch.abs(render_of_bwd - target_of_bwd) * target_bwd_mask)/(torch.sum(target_bwd_mask) + 1e-8)

        # scene flow smoothness loss
        # 这一部分计算空间平滑约束，对应原论文L_reg的L_sp项
        sf_sm_loss = args.w_sm * (compute_sf_sm_loss(ret['raw_pts_ref'], 
                                                    ret['raw_pts_post'], 
                                                    H, W, focal) \
                                + compute_sf_sm_loss(ret['raw_pts_ref'], 
                                                    ret['raw_pts_prev'], 
                                                    H, W, focal))

        # scene flow least kinectic loss
        # 这一部分计算时间平滑约束，对应原论文L_reg的L_temp项
        sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_ref'], 
                                                    ret['raw_pts_post'], 
                                                    ret['raw_pts_prev'], 
                                                    H, W, focal)
        sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_ref'], 
                                                    ret['raw_pts_post'], 
                                                    ret['raw_pts_prev'], 
                                                    H, W, focal)
        entropy_loss = args.w_entropy * torch.mean(-ret['raw_blend_w'] * torch.log(ret['raw_blend_w'] + 1e-8))

        # # ======================================  two-frames chain loss ===============================
        if chain_bwd:
            sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_prev'], 
                                                          ret['raw_pts_ref'], 
                                                          ret['raw_pts_pp'], 
                                                          H, W, focal)

        else:
            sf_sm_loss += args.w_sm * compute_sf_lke_loss(ret['raw_pts_post'], 
                                                          ret['raw_pts_pp'], 
                                                          ret['raw_pts_ref'], 
                                                          H, W, focal)

        if chain_5frames:
            print('5 FRAME RENDER LOSS ADDED') 
            render_loss += compute_mse(ret['rgb_map_pp_dy'], 
                                       target_rgb, 
                                       weights_map_dd)
        
        if args.decay_tau:
            tau = max(0.5, args.tau / (2 ** divsor))
        else:
            tau = args.tau
        
        if args.decay_arg_w:
            w_arg = args.w_arg / (decay_rate ** divsor)
        else:
            w_arg = args.w_arg
        
        arg_loss = w_arg * sampling_argmax_loss(ret['weights_fg'], -target_depth, 
                                                args.num_samples, tau, args.basis_type)
        
        if args.decay_gradient_w:
            w_gradient = args.w_gradient / (decay_rate ** divsor)
        else:
            w_gradient = args.w_gradient
        gradient_loss = w_gradient * compute_gradient_loss(ret['depth_map_ref'], -target_depth)

        w_empty = 0.1 / (decay_rate ** divsor)
        empty_loss = w_empty * compute_empty_loss(ret['z_vals'], ret['sigma_ref_dy'], -target_depth, 1 - target_mask, near, far)
        empty_loss += w_empty * compute_empty_loss(ret['z_vals'], ret['sigma_rigid'], -target_depth, target_mask, near, far)



        loss = sf_reg_loss + sf_cycle_loss + \
               render_loss + flow_loss + \
               sf_sm_loss + prob_reg_loss + \
               depth_loss + entropy_loss + \
               arg_loss + gradient_loss  + empty_loss
            #    gradient_loss  + empty_loss
               

        print('render_loss ', render_loss.item(), 
              ' bidirection_loss ', sf_cycle_loss.item(), 
              ' sf_reg_loss ', sf_reg_loss.item())
        print('depth_loss ', depth_loss.item(), 
              ' flow_loss ', flow_loss.item(), 
              ' sf_sm_loss ', sf_sm_loss.item())
        print('prob_reg_loss ', prob_reg_loss.item(),
              ' entropy_loss ', entropy_loss.item())
        print('arg_loss ', arg_loss.item(), 
              ' gradient_loss ', gradient_loss.item(), 
              ' empty_loss ', empty_loss.item())
        # print('gradient_loss ', gradient_loss.item(), 
        #       ' empty_loss ', empty_loss.item())
        
        loss.backward()
        optimizer.step()
        if args.learn_poses:
            optimizer_pose.step()
            scheduler_pose.step()
        if args.learn_focal:
            optimizer_focal.step()
            scheduler_focal.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))

            if args.N_importance > 0:
                if args.learn_poses:
                    torch.save({
                        'global_step': global_step,
                        'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                        'network_rigid': render_kwargs_train['network_rigid'].state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'poses_net_state_dict': pose_param_net.state_dict(),
                        'poses_optimizer_state_dict': optimizer_pose.state_dict(),
                        'focal_net_state_dict': focal_net.state_dict(), 
                        'focal_optimizer_state_dict': optimizer_focal.state_dict(), 
                    }, path)
                else:
                    torch.save({
                        'global_step': global_step,
                        'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                        'network_rigid': render_kwargs_train['network_rigid'].state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
            
            else:
                if args.learn_poses:
                    torch.save({
                        'global_step': global_step,
                        'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                        'network_rigid': render_kwargs_train['network_rigid'].state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'poses_net_state_dict': pose_param_net.state_dict(),
                        'poses_optimizer_state_dict': optimizer_pose.state_dict(),
                        'focal_net_state_dict': focal_net.state_dict(), 
                        'focal_optimizer_state_dict': optimizer_focal.state_dict(), 
                    }, path)
                else:
                    torch.save({
                        'global_step': global_step,
                        'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                        'network_rigid': render_kwargs_train['network_rigid'].state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)

            print('Saved checkpoints at', path)


        if i % args.i_print == 0 and i > 0:
            writer.add_scalar("train/loss", loss.item(), i)
            
            writer.add_scalar("train/render_loss", render_loss.item(), i)
            writer.add_scalar("train/depth_loss", depth_loss.item(), i)
            writer.add_scalar("train/flow_loss", flow_loss.item(), i)
            writer.add_scalar("train/prob_reg_loss", prob_reg_loss.item(), i)

            writer.add_scalar("train/sf_reg_loss", sf_reg_loss.item(), i)
            writer.add_scalar("train/sf_cycle_loss", sf_cycle_loss.item(), i)
            writer.add_scalar("train/sf_sm_loss", sf_sm_loss.item(), i)
            writer.add_scalar("train/arg_loss", arg_loss.item(), i)
            writer.add_scalar("train/gradient_loss", gradient_loss.item(), i)
            writer.add_scalar("train/empty_loss", empty_loss.item(), i)


        if i%args.i_img == 0:
            # img_i = np.random.choice(i_val)
            target = images[img_i]
            # pose = poses[img_i, :3,:4]
            target_depth = depths[img_i] - torch.min(depths[img_i])

            # img_idx_embed = img_i/num_img * 2. - 1.0

            # if img_i == 0:
            #     flow_fwd, fwd_mask = read_optical_flow(args.datadir, img_i, 
            #                                            args.start_frame, fwd=True)
            #     flow_bwd, bwd_mask = np.zeros_like(flow_fwd), np.zeros_like(fwd_mask)
            # elif img_i == num_img - 1:
            #     flow_bwd, bwd_mask = read_optical_flow(args.datadir, img_i, 
            #                                            args.start_frame, fwd=False)
            #     flow_fwd, fwd_mask = np.zeros_like(flow_bwd), np.zeros_like(bwd_mask)
            # else:
            #     flow_fwd, fwd_mask = read_optical_flow(args.datadir, 
            #                                            img_i, args.start_frame, 
            #                                            fwd=True)
            #     flow_bwd, bwd_mask = read_optical_flow(args.datadir, 
            #                                            img_i, args.start_frame, 
            #                                            fwd=False)

            # flow_fwd_rgb = torch.Tensor(flow_to_image(flow_fwd)/255.)#.cuda()
            # writer.add_image("val/gt_flow_fwd", 
            #                 flow_fwd_rgb, global_step=i, dataformats='HWC')
            # flow_bwd_rgb = torch.Tensor(flow_to_image(flow_bwd)/255.)#.cuda()
            # writer.add_image("val/gt_flow_bwd", 
            #                 flow_bwd_rgb, global_step=i, dataformats='HWC')

            with torch.no_grad():
                if args.learn_poses:
                    pose = pose_param_net(img_i)
                else:
                    pose = poses[img_i, :3,:4]
                ret = render(img_idx_embed, 
                             chain_bwd, False,
                             num_img, H, W, focal, 
                             chunk=1024*16, 
                             c2w=pose,
                             **render_kwargs_test)

                # pose_post = poses[min(img_i + 1, int(num_img) - 1), :3,:4]
                # pose_prev = poses[max(img_i - 1, 0), :3,:4]
                # render_of_fwd, render_of_bwd = compute_optical_flow(pose_post, pose, pose_prev, 
                #                                                     H, W, focal, ret, n_dim=2)

                # render_flow_fwd_rgb = torch.Tensor(flow_to_image(render_of_fwd.cpu().numpy())/255.)#.cuda()
                # render_flow_bwd_rgb = torch.Tensor(flow_to_image(render_of_bwd.cpu().numpy())/255.)#.cuda()
                
                writer.add_image("val/rgb_map_ref", torch.clamp(ret['rgb_map_ref'], 0., 1.), 
                                global_step=i, dataformats='HWC')
                writer.add_image("val/depth_map_ref", normalize_depth(ret['depth_map_ref']), 
                                global_step=i, dataformats='HW')

                writer.add_image("val/rgb_map_rig", torch.clamp(ret['rgb_map_rig'], 0., 1.), 
                                global_step=i, dataformats='HWC')
                writer.add_image("val/depth_map_rig", normalize_depth(ret['depth_map_rig']), 
                                global_step=i, dataformats='HW')

                writer.add_image("val/rgb_map_ref_dy", torch.clamp(ret['rgb_map_ref_dy'], 0., 1.), 
                                global_step=i, dataformats='HWC')
                writer.add_image("val/depth_map_ref_dy", normalize_depth(ret['depth_map_ref_dy']), 
                                global_step=i, dataformats='HW')

                # writer.add_image("val/rgb_map_pp_dy", torch.clamp(ret['rgb_map_pp_dy'], 0., 1.), 
                                # global_step=i, dataformats='HWC')

                writer.add_image("val/gt_rgb", target, 
                                global_step=i, dataformats='HWC')
                writer.add_image("val/monocular_disp", 
                                torch.clamp(target_depth /percentile(target_depth, 97), 0., 1.), 
                                global_step=i, dataformats='HW')

                writer.add_image("val/weights_map_dd", 
                                 ret['weights_map_dd'], 
                                 global_step=i, 
                                 dataformats='HW')

                if args.learn_poses:
                    store_current_pose(pose_param_net, pose_history_dir, i)

            # torch.cuda.empty_cache()

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()