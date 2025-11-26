
import sys
import os
import paths
import trimesh
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Diff3D'))
from Diff3D.diffusion import init_pipe
from Diff3D.dino import init_dino
from Diff3D.dataloaders.mesh_container import MeshContainer

source_dir = paths.source_folder_surreal5k
save_dir = paths.target_folder_surreal5k
device = 'cuda'
seed = 0
NUM = 49

os.makedirs(source_dir, exist_ok=True)
seed_everything(seed)

with torch.no_grad():
    paths = [os.path.join(source_dir, f) for f in sorted_nicely(os.listdir(source_dir))][:NUM]
    # Generating SMPL meshes and videos and jacobians
    pipe = init_pipe(device)
    dino_model = init_dino(device)
    for path in tqdm(paths):
        identity = path.split('/')[-1].replace('.off', '')

        os.makedirs(os.path.join(save_dir, identity), exist_ok=True)
        mesh_path = os.path.join(save_dir, identity, f"0.obj")
        
        mesh = trimesh.load_mesh(path, process=False)
        verts = np.asarray(mesh.vertices)       

        mesh.vertices = verts
        mesh.export(mesh_path)

        # Feature extraction
        source_mesh = MeshContainer().load_from_file(mesh_path)
        vertex_features = compute_features(device, pipe, dino_model, source_mesh, prompt='naked human')
        feat_path = os.path.join(save_dir, identity, f"features_0.pt")
        torch.save(vertex_features, feat_path)
