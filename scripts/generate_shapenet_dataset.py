import sys
import os
import open3d as o3d
from tqdm import tqdm
import trimesh
import torch
import paths

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Diff3D'))
from Diff3D.diffusion import init_pipe
from Diff3D.dino import init_dino
from Diff3D.dataloaders.mesh_container import MeshContainer


source_path = paths.source_folder_shapenet
save_dir = paths.target_folder_shapenet
categories = ['chair', 'airplane']
num_samples = 60
max_verts = 10000
skip_meshes = False

os.makedirs(save_dir, exist_ok=True)
synsets = {'04379243': 'table', '02958343': 'car', '03001627': 'chair', '02691156': 'airplane', '04256520': 'sofa', '04090263': 'rifle', '03636649': 'lamp', '04530566': 'watercraft', '02828884': 'bench', '03691459': 'loudspeaker', '02933112': 'cabinet', '03211117': 'display', '04401088': 'telephone', '02924116': 'bus', '02808440': 'bathtub', '03467517': 'guitar', '03325088': 'faucet', '03046257': 'clock', '03991062': 'flowerpot', '03593526': 'jar', '02876657': 'bottle', '02871439': 'bookshelf', '03642806': 'laptop', '03624134': 'knife', '04468005': 'train', '02747177': 'trash bin', '03790512': 'motorbike', '03948459': 'pistol', '03337140': 'file cabinet', '02818832': 'bed', '03928116': 'piano', '04330267': 'stove', '03797390': 'mug', '02880940': 'bowl', '04554684': 'washer', '04004475': 'printer', '03513137': 'helmet', '03761084': 'microwaves', '04225987': 'skateboard', '04460130': 'tower', '02942699': 'camera', '02801938': 'basket', '02946921': 'can', '03938244': 'pillow', '03710193': 'mailbox', '03207941': 'dishwasher', '04099429': 'rocket', '02773838': 'bag', '02843684': 'birdhouse', '03261776': 'earphone', '03759954': 'microphone', '04074963': 'remote', '03085013': 'keyboard', '02834778': 'bicycle', '02954340': 'cap', '02858304': 'boat', '02992529': 'mobile phone'}


o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
print('Number of categories', len(categories))

if not skip_meshes:
    for folder in tqdm(os.listdir(source_path)):
        category = synsets[folder]
        if not category in categories:
            print("Skipping:", category)
            continue

        category_folder = os.path.join(source_path, folder)
        items = sorted_nicely(os.listdir(category_folder))

        successes = 0
        for item in tqdm(items):
            
            mesh_path = os.path.join(category_folder, item, 'models', 'model_normalized.obj')
            if not os.path.exists(mesh_path):
                continue

            mesh = trimesh.load_mesh(mesh_path, process=False, skip_materials=True)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

            dist_folder = os.path.join(save_dir, f'{category}_{item}')
            os.makedirs(dist_folder, exist_ok=True)
            
            remeshed = refine_mesh(mesh, max_verts)
            remeshed.export(os.path.join(dist_folder, '0.obj'))
            tqdm.write(f"{category}, {item}")

            successes += 1
            if successes == num_samples:
                break

        if successes < num_samples:
            print("Could not find enough samples for", category)

with torch.no_grad():
    device = 'cuda'
    pipe = init_pipe(device)
    dino_model = init_dino(device)
    for folder in tqdm(os.listdir(save_dir)):
        tqdm.write(f"Working on {folder}")
        category = folder.split('_')[0]
        mesh_dist_path = os.path.join(save_dir, folder, '0.obj')
        source_mesh = MeshContainer().load_from_file(mesh_dist_path)
        vertex_features = compute_features(device, pipe, dino_model, source_mesh, prompt=category)
        feat_path = os.path.join(save_dir, folder, "features_0.pt")
        torch.save(vertex_features, feat_path)
