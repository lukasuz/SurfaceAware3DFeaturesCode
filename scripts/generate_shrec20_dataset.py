import sys
import os
from tqdm import tqdm
import shutil
from scipy.io import loadmat
import numpy as np
import trimesh
import paths

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Diff3D'))
from Diff3D.diffusion import init_pipe
from Diff3D.dino import init_dino
from Diff3D.dataloaders.mesh_container import MeshContainer

shrec_path = paths.source_folder_shrec20
tgt_dir = paths.target_folder_shrec20
gt_maps = paths.gt_maps_shrec20
device = 'cuda'
seed = 0

# Load each test setup
test_sets = {}
test_sets_path = os.path.join(shrec_path, "test-sets")
for ts in os.listdir(test_sets_path):
    with open(os.path.join(test_sets_path, ts), 'r') as f:
        lines = f.readlines()
        for line in lines:
            src, tgt = line.split(',')
            tgt = tgt.replace('\n', '')
            if src not in test_sets:
                test_sets[src] = []
            test_sets[src].append(tgt)

# For each test setup, find the overlapping test points and save the index positions
# get overlapping test points
# for each point get index vertices of each file
for src in test_sets.keys():
    src_path = os.path.join(tgt_dir, src)
    os.makedirs(src_path, exist_ok=True)
    shutil.copy(os.path.join(shrec_path, "models", f"{src}.obj"), os.path.join(src_path,  f"0.obj"))

    src_mat = loadmat(os.path.join(gt_maps, f"{src}.mat"))

    src_mesh = trimesh.load(os.path.join(shrec_path, "models", f"{src}.obj"), process=False)
    src_verts = np.asarray(src_mesh.vertices)
    for tgt in test_sets[src]:
        tgt_mat = loadmat(os.path.join(gt_maps, f"{tgt}.mat"))
        tgt_mesh = trimesh.load(os.path.join(shrec_path, "models", f"{tgt}.obj"), process=False)
        tgt_verts = np.asarray(tgt_mesh.vertices)
    
        matching_points = []
        for si, p in enumerate(src_mat['points'][:,0]):
            src_vert_i = np.argmin(np.linalg.norm((src_mat['centroids'][si] - src_verts), axis=1))
            print('diff src:', src_vert_i - src_mat['verts'][si].item())
            if p in tgt_mat['points'][:,0]:
                ti = np.where(tgt_mat['points'][:,0] == p)[0]
                tgt_vert_i = np.argmin(np.linalg.norm((tgt_mat['centroids'][ti] - tgt_verts), axis=1))
                print('diff tgt:', tgt_vert_i - tgt_mat['verts'][ti].item())
                matching_points.append([src_vert_i, tgt_vert_i])
                
                # matching_points.append([src_mat['centroids'][si], tgt_mat['centroids'][ti.item()]])

        print(f"Found {len(matching_points)} between {src} and  {tgt}")
        map_path = os.path.join(src_path, f"{tgt}.npy")
        np.save(map_path, np.array(matching_points))

for p in os.listdir(os.path.join(shrec_path, 'models')):
    pp = os.path.join(os.path.join(shrec_path, 'models'), p)
    name = p.split('.')[0]

    src_path = os.path.join(tgt_dir, name)
    if not os.path.exists(os.path.join(src_path,  f"0.obj")):
        print('creating', src_path)
        os.makedirs(src_path, exist_ok=True)
        shutil.copy(os.path.join(shrec_path, "models", f"{src}.obj"), os.path.join(src_path,  f"0.obj"))

folders = os.listdir(tgt_dir)
seed_everything(seed)
all_thetas = []
with torch.no_grad():

    pipe = init_pipe(device)
    dino_model = init_dino(device)
    # Generating Diff3D features
    for file in tqdm(folders):
        mesh_path = os.path.join(tgt_dir, file, "0.obj")

        mesh = trimesh.load_mesh(mesh_path, process=False)
        mesh.export(mesh_path)

        source_mesh = MeshContainer().load_from_file(mesh_path)
        animal_name = file.split("_")[0]
        vertex_features = compute_features(device, pipe, dino_model, source_mesh, prompt=animal_name)
        feat_path = os.path.join(tgt_dir, file, f"features_0.pt")
        torch.save(vertex_features, feat_path)
