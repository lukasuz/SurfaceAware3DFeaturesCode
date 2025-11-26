import argparse
import json
import os

def save_args(args, path=''):
    with open(os.path.join(path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

def load_args(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            args = json.load(f)
        return argparse.Namespace(**args)
    else:
        return None

def get_args(verbose=True):
    parser = argparse.ArgumentParser(description='Default argparse arguments for the project.')

    # General args
    parser.add_argument('--feature_dim', type=int, default=2048, help='Input feature dim (Diff3D features)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--exp_path', type=str, default='./experiments/train', help='root folder for experiments')
    parser.add_argument('--exp_name', type=str, default='test', help='Name for experiment folder, otherwise will be created automatically')
    parser.add_argument('--FN_path', type=str, default="", help='path to feature network, otherwise the one in the same folder is used')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')

    # Paths
    parser.add_argument('--smpl_data_path', type=str, default='./data/SMPL_shape_dataset', help='Path to SMPL data')
    parser.add_argument('--amass_path', type=str, default='PATH/TO/AMASS/amass_30fps_train.pt', help='Path to AMASS data')
    parser.add_argument('--smplh_path', type=str, default='PATH/TO/smplh/neutral/model.npz', help='Path to smplh model')
    parser.add_argument('--deforming_things_path', type=str, default='data/DeformingThings4DFeatures', help='')
    parser.add_argument('--smal_ours_data_path', type=str, default='./data/SMAL_ours_shape_dataset', help='')
    parser.add_argument('--shrec19_data_path', type=str, default='./data/SHREC19_shape_dataset', help='')
    parser.add_argument('--shrec20_data_path', type=str, default='./data/SHREC20_shape_dataset', help='')
    parser.add_argument('--tosca_data_path', type=str, default='./data/TOSCA_shape_dataset', help='')
    parser.add_argument('--shapnet_chair_data_path', type=str, default='./data/shapenet_chair', help='')
    parser.add_argument('--shapnet_chair_val_data_path', type=str, default='./data/shapenet_chair_val', help='')
    parser.add_argument('--shapnet_airplane_data_path', type=str, default='./data/shapenet_airplane', help='')
    parser.add_argument('--shapnet_airplane_val_data_path', type=str, default='./data/shapenet_airplane_val', help='')
    parser.add_argument('--shapnet_table_data_path', type=str, default='./data/shapenet_table', help='')
    parser.add_argument('--smal_data_path', type=str, default='./data/SMAL_shape_dataset', help='')
    parser.add_argument('--surreal_data_path', type=str, default='./data/surreal_shape_dataset', help='')
    parser.add_argument('--shapenet_data_path', type=str, default='./data/shapenet', help='')
    parser.add_argument('--polyhaven_chair_data_path', type=str, default='./data/polyhaven_chairs', help='')
    parser.add_argument('--polyhaven_animals_data_path', type=str, default='./data/polyhaven_animals', help='')
    parser.add_argument('--source_folder_dt4d', type=str, default='/PATH/TO/DeformingThings4D/animals', help='Path to DeformingThings4D data')

    parser.add_argument("--mask", nargs="+", type=int, default=None)
    parser.add_argument("--train_data", nargs="+", type=str)
    parser.add_argument("--val_pairs", nargs="+", type=str, default=[])
    parser.add_argument('--tensorboard_path', type=str, default='', help='Path to TensorBoard logs')
    parser.add_argument('--train_on_validation', action='store_true', help='Whether to include the validation set in training')

    # Data sampling
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--feature_noise_p', type=float, default=0., help='Feature noise probability')
    parser.add_argument('--fps_p', type=int, default=100, help='FPS parameter')
    # Feature Network
    parser.add_argument('--FN_lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--w_contrastive', type=float, default=1, help='Weight for geodesic loss')
    parser.add_argument('--w_reconstruction', type=float, default=1, help='Weight for reconstruction loss')
    parser.add_argument('--FN_iters', type=int, default=60000, help='Number of iterations')
    parser.add_argument('--FN_render_iter', type=int, default=500, help='Print iteration interval')
    parser.add_argument('--FN_train', type=bool, default=True, help='Train flag')

    args =  parser.parse_args()

    args_path = os.path.join(args.exp_path, args.exp_name, 'args.json')
    if os.path.exists(args_path):
        print("Loaded arguments from", args_path)
        args = load_args(args_path)
    else:
        os.makedirs(os.path.join(args.exp_path, args.exp_name), exist_ok=True)
        save_args(args, os.path.join(args.exp_path, args.exp_name))

    if verbose:
        print("Arguments:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")

    return args

if __name__ == '__main__':
    args = get_args()
    print(args)
    save_args(args)
    args = load_args()
    print(args)
