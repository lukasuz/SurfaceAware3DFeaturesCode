import sys
import os
import torch
from tqdm import tqdm
import numpy as np
import open3d as o3d
import torch
import re 
from contextlib import redirect_stdout, redirect_stderr
import warnings
import paths

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Diff3D'))
from Diff3D.diffusion import init_pipe
from Diff3D.dino import init_dino
from Diff3D.dataloaders.mesh_container import MeshContainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import sorted_nicely, anime_read, compute_features

source_folder = paths.source_folder_dt4d
target_folder = paths.target_folder_dt4d
os.makedirs(target_folder, exist_ok=True)

device = torch.device('cuda:0')
seed = 0
NUM_SEQ = 5
SKIP_MESHES = False

def find_first_index(s):
    match = re.search(r'[A-Z0-9]', s)
    return match.start() if match else -1

def prune_mesh(verts, faces,  visible_faces):
    new_verts = []
    _faces = faces[visible_faces]
    v_map = np.array([None] * len(verts))

    # Iterate over vertice to preserve order
    v_mask = np.zeros(len(verts), dtype=bool)
    for i, vertex in enumerate(verts):
        if i in _faces:
            v_mask[i] = True
            v_map[i] = len(new_verts)
            new_verts.append(vertex)
    new_verts = np.stack(new_verts)
    # Exchange old vertex indices with new ones
    # sanity check : np.unique((v_map[faces] == None).astype(int).sum(axis=-1))
    new_faces = v_map[_faces].astype(np.int32)
    assert np.all(np.unique((new_faces == None).astype(int).sum(axis=-1)) == 0)

    return new_verts, new_faces, v_mask

def mute(func, *args, **kwargs):
    with open(os.devnull, 'w') as fnull, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(fnull), redirect_stderr(fnull):
            return func(*args, **kwargs)

torch.cuda.set_device(device)
all_sequences = os.listdir(source_folder)
all_sequences = sorted_nicely(all_sequences)
identities = [seq.split('_')[0] for seq in all_sequences]
identities = list(set(identities))
identities = sorted_nicely(identities)
sequenes_per_identity = [[seq for seq in all_sequences if identity in seq] for identity in identities]

# Choose 5 sequences per identity
np.random.seed(seed)
choosen_sequences = []
for seqs in sequenes_per_identity:
    _len = len(seqs)
    if _len >= NUM_SEQ:
        choosen = list(np.random.choice(seqs, NUM_SEQ, replace=False))
    else:
        _diff = NUM_SEQ - _len
        choosen = seqs + list(np.random.choice(seqs, _diff, replace=True))

    choosen_sequences.append(choosen)

# Choose meshes
if not SKIP_MESHES:
    for identity, sequences in zip(identities, choosen_sequences):
        print("Identity:", identity)
        print("Choosen sequences:", sequences)

        folder_path = os.path.join(target_folder, identity)
        os.makedirs(folder_path, exist_ok=True)

        all_faces = []
        all_vertices = []
        all_vmasks = []
        rnd_ts = []
        for s, seq in enumerate(sequences):
            # Load Mesh
            seq_path = os.path.join(source_folder, seq, seq + ".anime") 
            nf, nv, nt, vertices, faces, offset_data = anime_read(seq_path)

            rnd_t = np.random.choice(len(offset_data), 1)[0]

            folder_path = os.path.join(target_folder, identity)
            v_mask_path = os.path.join(folder_path, 'v_mask0.npy')
            faces_path = os.path.join(folder_path, 'faces0.npy')
            # save faces and v mask after pruning for biggest component (or load)
            os.makedirs(folder_path, exist_ok=True)
            if s == 0:
                o3d_faces = o3d.utility.Vector3iVector(faces)
                o3d_verts = o3d.utility.Vector3dVector(vertices)
                mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)

                triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
                max_i = np.argmax(cluster_n_triangles)
                visible_faces = np.where(np.array(triangle_clusters) == max_i)[0]

                vertices, faces, v_mask = prune_mesh(vertices, faces, visible_faces)

                np.save(v_mask_path, v_mask)
                np.save(faces_path, faces)
            else:
                v_mask = np.load(v_mask_path)
                faces = np.load(faces_path)
                vertices = vertices[v_mask]

                os.path.join(target_folder, identity)

            # Apply random deformation from sequence
            vertices = vertices + offset_data[rnd_t][v_mask]
            rnd_ts.append(rnd_t)

            o3d_faces = o3d.utility.Vector3iVector(faces)
            o3d_verts = o3d.utility.Vector3dVector(vertices)
            mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)

            assert np.all(vertices == np.asarray(mesh.vertices)), "vertex order was changed"
            assert np.all(faces == np.asarray(mesh.triangles)), "face order was changed"
            
            o3d.io.write_triangle_mesh(os.path.join(folder_path, f"{s}.obj"), mesh)

        with open(os.path.join(folder_path, 'setup.txt'), 'w') as f:
            f.write(f"identity,sequence,time\n")
            for i in range(len(sequences)):
                f.write(f"{identity},{sequences[i]},{rnd_ts[i]}\n")

pipe = init_pipe(device)
dino_model = init_dino(device)

# Extract Features
for i, (identity, sequences) in enumerate(zip(identities, choosen_sequences)):
    tqdm.write(f"Identity: {identity} ({i}/{len(identities)})")

    folder_path = os.path.join(target_folder, identity)
    for s, seq in tqdm(enumerate(sequences)):
        # Load Mesh
        feature_path = os.path.join(folder_path, f"features_{s}.pt")

        if not os.path.exists(feature_path):
            try:
                idx = find_first_index(seq)
                if idx == -1:
                    animal_str = 'animal'
                else:
                    animal_str = seq[:find_first_index(seq)]

                mesh_path = os.path.join(folder_path, f"{s}.obj")
                mesh = MeshContainer().load_from_file(mesh_path)
                features = compute_features(device, pipe, dino_model, mesh, animal_str)
                torch.save(features, feature_path)
            except Exception as e:
                # append to error.log
                with open(os.path.join(target_folder, 'error.log'), 'a') as f:
                    # time string
                    f.write(f"Error for sequence: {seq}\n")
                    f.write(f"{e}\n")
                    f.write("\n")
                    f.write("\n")
