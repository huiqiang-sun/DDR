import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import draw

from render_utils import *
from run_nerf_helpers import *
from load_llff import *
from poses import *
from focal import *

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)

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

    parser.add_argument("--target_idx", type=int, default=7, 
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
    parser.add_argument("--tau", type=float, default=2., help='temperature')
    parser.add_argument("--decay_tau", action='store_true', help='temperature decay')

    parser.add_argument("--w_gradient", type=float, default=0.01, help='w_gradient regularization weight')
    parser.add_argument("--decay_gradient_w", action='store_true', help='decay gradient weights')

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

    return parser


def main():
    parser = config_parser()
    args = parser.parse_args()

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

        # c2w = poses[target_idx, :, :]
        # render_poses = render_wander_path(c2w)
        # render_poses = np.array(render_poses).astype(np.float32)
        
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
    

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    basedir = args.basedir
    args.expname = args.expname + '_F%02d-%02d'%(args.start_frame, args.end_frame)
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

    # Create NeRF model
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

        if args.ft_path is not None and args.ft_path!='None':
            ckpts = [args.ft_path]
        else:
            ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]

            print('Reloading poses params from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            pose_param_net.load_state_dict(ckpt['poses_net_state_dict'])
    
    if args.learn_focal:
        focal_net = LearnFocal(init_focal=torch.tensor(focal))
        device_ids = list(range(torch.cuda.device_count()))
        focal_net = torch.nn.DataParallel(focal_net, device_ids=device_ids)

        if args.ft_path is not None and args.ft_path!='None':
            ckpts = [args.ft_path]
        else:
            ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]

            print('Reloading focal params from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            focal_net.load_state_dict(ckpt['focal_net_state_dict'])

    
    images = torch.Tensor(images).to(device)
    depths = torch.Tensor(depths).to(device)
    masks = 1.0 - torch.Tensor(masks).to(device)
    poses = torch.Tensor(poses).to(device)
    with torch.no_grad():
        hwf_temp = ref_c2w[:,4:5]
        if args.learn_focal:
            focal = focal_net()
            hwf = [H, W, focal]
            hwf_temp = torch.tensor(hwf).reshape([3,1])
            hwf_temp = hwf_temp.cpu().numpy()
        if args.learn_poses:
            c2w = pose_param_net(target_idx).cpu().numpy()  
            render_poses = render_wander_path(c2w, hwf=hwf_temp)
            render_poses = np.array(render_poses).astype(np.float32)
            render_poses = torch.Tensor(render_poses).to(device)
            pose = pose_param_net(target_idx)
        else:
            render_poses = torch.Tensor(render_poses).to(device)
            pose = poses[target_idx, :3,:4]
        c2w = render_poses[12]
        c2w = c2w[:3, :4]

    ################################################
    target = images[target_idx]
    target_depth = depths[target_idx] - torch.min(depths[target_idx])

    save_img_dir = os.path.join(basedir, expname, 'test_images')
    os.makedirs(save_img_dir, exist_ok=True)

    num_img = float(images.shape[0])
    img_idx_embed = target_idx/num_img * 2. - 1.0
    chain_5frames = False

    with torch.no_grad():
        ret = render(img_idx_embed, 
                     0, 
                     chain_5frames,
                     num_img, H, W, focal, 
                     chunk=1024*16, 
                     c2w=pose,
                     **render_kwargs_test)

    weights_fg = ret['weights_fg'].cpu().numpy()
    weights_dy = ret['weights_dy'].cpu().numpy()
    weights_rig = ret['weights_rig'].cpu().numpy()
    Ts = ret['Ts'].cpu().numpy()
    alpha_dy = ret['opacity_dy'].cpu().numpy()
    alpha_rig = ret['opacity_rig'].cpu().numpy()

    # 打印最终渲染的rgb图和深度图
    depth = torch.clamp(ret['depth_map_ref'] / percentile(ret['depth_map_ref'], 97), 0., 1.)
    rgb = ret['rgb_map_ref'].cpu().numpy()

    rr, cc = draw.line(150, 0, 150, W-1)
    rgb[rr, cc, :] = [255, 0, 0]


    if save_img_dir is not None:
        rgb8 = to8b(rgb)
        depth8 = to8b(depth.unsqueeze(-1).repeat(1, 1, 3).cpu().numpy())

        start_y = (rgb8.shape[1] - W) // 2
        rgb8 = rgb8[:, start_y:start_y+ W, :]
        depth8 = depth8[:, start_y:start_y+ W, :]
        # print(rgb8.shape)
        # rgb_cut = rgb8[128:161, 128:144]

        filename = os.path.join(save_img_dir, 'rgb.jpg')
        imageio.imwrite(filename, rgb8)
        filename = os.path.join(save_img_dir, 'depth.jpg')
        imageio.imwrite(filename, depth8)

        # filename = os.path.join(save_img_dir, 'rgb_cut.jpg')
        # imageio.imwrite(filename, rgb_cut)
    
    # 打印最终渲染的动态rgb图和动态深度图
    depth = torch.clamp(ret['depth_map_ref_dy'] / percentile(ret['depth_map_ref_dy'], 97), 0., 1.)
    rgb = ret['rgb_map_ref_dy'].cpu().numpy()

    if save_img_dir is not None:
        rgb8 = to8b(rgb)
        depth8 = to8b(depth.unsqueeze(-1).repeat(1, 1, 3).cpu().numpy())

        start_y = (rgb8.shape[1] - W) // 2
        rgb8 = rgb8[:, start_y:start_y+ W, :]
        depth8 = depth8[:, start_y:start_y+ W, :]

        filename = os.path.join(save_img_dir, 'rgb_fg.jpg')
        imageio.imwrite(filename, rgb8)
        filename = os.path.join(save_img_dir, 'depth_fg.jpg')
        imageio.imwrite(filename, depth8)

    
    # 打印最终渲染的静态rgb图和静态深度图
    depth = torch.clamp(ret['depth_map_rig'] / percentile(ret['depth_map_rig'], 97), 0., 1.)
    rgb = ret['rgb_map_rig'].cpu().numpy()

    if save_img_dir is not None:
        rgb8 = to8b(rgb)
        depth8 = to8b(depth.unsqueeze(-1).repeat(1, 1, 3).cpu().numpy())

        start_y = (rgb8.shape[1] - W) // 2
        rgb8 = rgb8[:, start_y:start_y+ W, :]
        depth8 = depth8[:, start_y:start_y+ W, :]

        filename = os.path.join(save_img_dir, 'rgb_rig.jpg')
        imageio.imwrite(filename, rgb8)
        filename = os.path.join(save_img_dir, 'depth_rig.jpg')
        imageio.imwrite(filename, depth8)

    print(weights_dy.shape)
    print('Ts: ', Ts.shape)
    print('alpha_dy: ', alpha_dy.shape)
    print('alpha_rig: ', alpha_rig.shape)
    weights_dy = weights_dy[150,:]
    weights_rig = weights_rig[150,:]
    weights_fg = weights_fg[150,:]
    weights_mix = weights_dy + weights_rig
    Ts = Ts[150,:]
    alpha_dy = alpha_dy[150,:]
    alpha_rig = alpha_rig[150,:]

    temp = weights_fg * alpha_dy

    cmap = plt.cm.get_cmap('jet')
    weights_rig_map = cmap(weights_rig / (weights_rig.max()+1e-10)) * 255
    weights_rig_map = weights_rig_map[:, :, 0:3].astype(np.uint8)
    weights_dy_map = cmap(weights_dy / (weights_dy.max()+1e-10)) * 255
    weights_dy_map = weights_dy_map[:, :, 0:3].astype(np.uint8)
    weights_fg_map = cmap(weights_fg / (weights_fg.max()+1e-10)) * 255
    weights_fg_map = weights_fg_map[:, :, 0:3].astype(np.uint8)
    weights_mix_map = cmap(weights_mix / (weights_mix.max()+1e-10)) * 255
    weights_mix_map = weights_mix_map[:, :, 0:3].astype(np.uint8)
    Ts_map = cmap(Ts / (Ts.max()+1e-10)) * 255
    Ts_map = Ts_map[:, :, 0:3].astype(np.uint8)
    alpha_dy_map = cmap(alpha_dy / (alpha_dy.max()+1e-10)) * 255
    alpha_dy_map = alpha_dy_map[:, :, 0:3].astype(np.uint8)
    alpha_rig_map = cmap(alpha_rig / (alpha_rig.max()+1e-10)) * 255
    alpha_rig_map = alpha_rig_map[:, :, 0:3].astype(np.uint8)
    temp_map = cmap(temp / (temp.max()+1e-10)) * 255
    temp_map = temp_map[:, :, 0:3].astype(np.uint8)
    print(weights_rig_map.shape)
    filename1 = os.path.join(save_img_dir, 'weights_rig.jpg')
    filename2 = os.path.join(save_img_dir, 'weights_dy.jpg')
    filename3 = os.path.join(save_img_dir, 'weights_fg.jpg')
    filename4 = os.path.join(save_img_dir, 'weights_mix.jpg')
    filename5 = os.path.join(save_img_dir, 'Ts.jpg')
    filename6 = os.path.join(save_img_dir, 'alpha_dy.jpg')
    filename7 = os.path.join(save_img_dir, 'alpha_rig.jpg')
    filename8 = os.path.join(save_img_dir, 'temp.jpg')

    fig = plt.figure(dpi=800)
    ax1 = fig.add_subplot(1 ,1, 1)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.imshow(weights_rig_map[:W, ...])
    plt.savefig(filename1)

    ax2 = fig.add_subplot(1 ,1, 1)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.imshow(weights_dy_map[:W, ...])
    plt.savefig(filename2)

    ax3 = fig.add_subplot(1 ,1, 1)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.imshow(weights_fg_map[:W, ...])
    plt.savefig(filename3)

    ax4 = fig.add_subplot(1 ,1, 1)
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax4.imshow(weights_mix_map[:W, ...])
    plt.savefig(filename4)

    ax5 = fig.add_subplot(1 ,1, 1)
    ax5.get_xaxis().set_visible(False)
    ax5.get_yaxis().set_visible(False)
    ax5.imshow(Ts_map[:W, ...])
    plt.savefig(filename5)

    ax6 = fig.add_subplot(1 ,1, 1)
    ax6.get_xaxis().set_visible(False)
    ax6.get_yaxis().set_visible(False)
    ax6.imshow(alpha_dy_map[:W, ...])
    plt.savefig(filename6)

    ax7 = fig.add_subplot(1 ,1, 1)
    ax7.get_xaxis().set_visible(False)
    ax7.get_yaxis().set_visible(False)
    ax7.imshow(alpha_rig_map[:W, ...])
    plt.savefig(filename7)

    ax8 = fig.add_subplot(1 ,1, 1)
    ax8.get_xaxis().set_visible(False)
    ax8.get_yaxis().set_visible(False)
    ax8.imshow(temp_map[:W, ...])
    plt.savefig(filename8)
    plt.close()
    # imageio.imwrite(filename, weights_dy_map)


def main2():
    parser = config_parser()
    args = parser.parse_args()

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
    

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    basedir = args.basedir
    args.expname = args.expname + '_F%02d-%02d'%(args.start_frame, args.end_frame)
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

    # Create NeRF model
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

        if args.ft_path is not None and args.ft_path!='None':
            ckpts = [args.ft_path]
        else:
            ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]

            print('Reloading poses params from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            pose_param_net.load_state_dict(ckpt['poses_net_state_dict'])
    
    if args.learn_focal:
        focal_net = LearnFocal(init_focal=torch.tensor(focal))
        device_ids = list(range(torch.cuda.device_count()))
        focal_net = torch.nn.DataParallel(focal_net, device_ids=device_ids)

        if args.ft_path is not None and args.ft_path!='None':
            ckpts = [args.ft_path]
        else:
            ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]

            print('Reloading focal params from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            focal_net.load_state_dict(ckpt['focal_net_state_dict'])

    
    images = torch.Tensor(images).to(device)
    depths = torch.Tensor(depths).to(device)
    masks = 1.0 - torch.Tensor(masks).to(device)
    poses = torch.Tensor(poses).to(device)
    with torch.no_grad():
        hwf_temp = ref_c2w[:,4:5]
        if args.learn_focal:
            focal = focal_net()
            hwf = [H, W, focal]
            hwf_temp = torch.tensor(hwf).reshape([3,1])
            hwf_temp = hwf_temp.cpu().numpy()
        if args.learn_poses:
            c2w = pose_param_net(target_idx).cpu().numpy()  
            render_poses = render_wander_path(c2w, hwf=hwf_temp)
            render_poses = np.array(render_poses).astype(np.float32)
            render_poses = torch.Tensor(render_poses).to(device)
            pose = pose_param_net(target_idx)
        else:
            render_poses = torch.Tensor(render_poses).to(device)
            pose = poses[target_idx, :3,:4]
        c2w = render_poses[12]
        c2w = c2w[:3, :4]

    ################################################
    target = images[target_idx]
    target_depth = depths[target_idx] - torch.min(depths[target_idx])

    save_img_dir = os.path.join(basedir, expname, 'test')
    os.makedirs(save_img_dir, exist_ok=True)

    num_img = float(images.shape[0])
    img_idx_embed = target_idx/num_img * 2. - 1.0
    chain_5frames = False

    with torch.no_grad():
        ret = render(img_idx_embed, 
                     0, 
                     chain_5frames,
                     num_img, H, W, focal, 
                     chunk=1024*16, 
                     c2w=pose,
                     **render_kwargs_test)

    weights_FG = ret['weights_fg'].cpu().numpy()
    weights_dy = ret['weights_dy'].cpu().numpy()
    weights_rig = ret['weights_rig'].cpu().numpy()
    Ts = ret['Ts'].cpu().numpy()
    alpha_dy = ret['opacity_dy'].cpu().numpy()
    alpha_rig = ret['opacity_rig'].cpu().numpy()

    # 打印最终渲染的rgb图和深度图
    depth = torch.clamp(ret['depth_map_ref'] / percentile(ret['depth_map_ref'], 97), 0., 1.)
    rgb = ret['rgb_map_ref'].cpu().numpy()

    a = 54
    b = 224
    rr, cc = draw.line(a, 0, a, 383)
    rgb[rr, cc, :] = [255, 0, 0]
    rr, cc = draw.line(b, 0, b, 383)
    rgb[rr, cc, :] = [255, 0, 0]

    if save_img_dir is not None:
        rgb8 = to8b(rgb)
        start_y = (rgb8.shape[1] - 384) // 2
        rgb8 = rgb8[:, start_y:start_y+ 384, :]

        filename = os.path.join(save_img_dir, 'rgb.jpg')
        imageio.imwrite(filename, rgb8)

    for i in range(a, b):
        weights_fg = weights_FG[i,:]
        cmap = plt.cm.get_cmap('jet')
        weights_fg_map = cmap(weights_fg / (weights_fg.max()+1e-10)) * 255
        weights_fg_map = weights_fg_map[:, :, 0:3].astype(np.uint8)
        filename = os.path.join(save_img_dir, '{:03d}.jpg'.format(i))
        fig = plt.figure(dpi=120)
        ax = fig.add_subplot(1 ,1, 1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.imshow(weights_fg_map[:384, ...])
        plt.savefig(filename)
        plt.close()

        print(i)





if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main()