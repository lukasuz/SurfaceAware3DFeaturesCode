
import sys
import os
from tqdm import tqdm
import numpy as np
import trimesh
from pathlib import Path
import re
import paths

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import *

from Diff3D.diffusion import init_pipe
from Diff3D.dino import init_dino
from Diff3D.dataloaders.mesh_container import MeshContainer

source_dir = paths.source_folder_tosca
save_dir = paths.target_folder_tosca
device = 'cuda'

with torch.no_grad():
    paths = [str(path) for path in Path(source_dir).rglob("*.off") if not path.name.startswith('.')]
    paths = list(filter(lambda file: 'victoria' not in file and 'david' not in file and 'michael' not in file, paths))
    # Generating SMPL meshes and videos and jacobians
    pipe = init_pipe(device)
    dino_model = init_dino(device)
    for path in tqdm(paths):
        identity = path.split('/')[-1].split('.')[0]
        animal_name = re.sub(r'\d+', '', identity)

        os.makedirs(os.path.join(save_dir, identity), exist_ok=True)
        mesh_path = os.path.join(save_dir, identity, f"0.obj")

        mesh = trimesh.load_mesh(path, process=False)
        verts = np.asarray(mesh.vertices)
        _verts = np.copy(verts)

        verts[:,0] = _verts[:,1] # flip so visualisation is consistent
        verts[:,1] = _verts[:,2]
        verts[:,2] = _verts[:,0]
        mesh.vertices = verts
        mesh.export(mesh_path)

        # Feature extraction
        feat_path = os.path.join(save_dir, identity, f"features_0.pt")
        source_mesh = MeshContainer().load_from_file(mesh_path)
        vertex_features = compute_features(device, pipe, dino_model, source_mesh, prompt=animal_name, is_tosca=True)
        torch.save(vertex_features, feat_path)
