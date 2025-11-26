from torch.utils.data import Dataset
import torch
import numpy as np
import joblib
import os
from utils import FPS_potpourri, sorted_nicely, SMALMesh, anime_read
import trimesh
from human_body_prior.body_model.body_model import BodyModel
from torch.utils.data import Dataset
import numpy as np


def sample_data_set(args, path, kfold, few_shot=False, num_feat_per_identity=1):
    assert type(kfold) == tuple and len(kfold) == 2
    assert kfold[0] < kfold[1]

    k, step = kfold
    num_identities = ShapeDataset.num_identites(path)
    assert num_identities % kfold[1] == 0, "number of samples must be divisible by number of folds"

    if num_feat_per_identity > 1:
        assert kfold[1] == num_identities, "kfold logic only applicable over identities"
        step = num_feat_per_identity
        indices = np.arange(num_identities * num_feat_per_identity)
    else:
        indices = np.arange(num_identities)
    indices = np.roll(indices, int(k * step))
    test_mask = indices[:step]
    train_val_mask = indices[step:]

    train_val_partition = ShapeDataset(path, num_feat_per_identity=num_feat_per_identity, mask=train_val_mask, **vars(args))
    test_partition = ShapeDataset(path, num_feat_per_identity=num_feat_per_identity, mask=test_mask, **vars(args))

    if few_shot:
        temp = train_val_partition
        train_val_partition = test_partition
        test_partition = temp

    print(f"Fold: {kfold}")
    print(f"Train/Val length: {len(train_val_partition)}")
    print(f"Test length: {len(test_partition)}")
    ShapeDataset.check_data_leakage(test_partition, train_val_partition)

    return train_val_partition, test_partition

def get_data_shrec19_shape(args):
    return ShapeDataset(args.shrec19_data_path, num_feat_per_identity=1, **vars(args))

def get_data_shrec20_shape(args):
    return ShapeDataset(args.shrec20_data_path, num_feat_per_identity=1, **vars(args))

def get_data_smpl_shape(args, kfold=None, few_shot=False):
    if args.mask is not None:
        assert kfold is None and not few_shot, "No splitting available when passing mask"
    with torch.no_grad():
        body_model = BodyModel(args.smplh_path, 'smplh', num_betas=10, batch_size=1)
        weights = body_model.weights
        # Ignore fingers, assign weights to parent
        lh_mask = torch.where(weights[:,22:37] > 0)[0]
        rh_mask = torch.where(weights[:,37:] > 0)[0]
        weights[rh_mask] = 0
        weights[rh_mask,21] = 1
        weights[lh_mask] = 0
        weights[lh_mask,20] = 1

    if kfold is not None:
        train_val_partition, test_partition = sample_data_set(args, args.smpl_data_path, kfold, few_shot, num_feat_per_identity=1)
        return train_val_partition, test_partition, weights.to(args.device)

    return ShapeDataset(args.smpl_data_path, num_feat_per_identity=1, **vars(args)), weights.to(args.device)

def get_data_smal_shape(args, **kwargs):
    return ShapeDataset(args.smal_data_path, num_feat_per_identity=1, **vars(args)), None

def get_data_surreal_shape(args, **kwargs):
    return ShapeDataset(args.surreal_data_path, num_feat_per_identity=1, **vars(args)), None

def get_data_shapenet(args, **kwargs):
    return ShapeDataset(args.shapenet_data_path, num_feat_per_identity=1, **vars(args)), None

def get_data_tosca(args, **kwargs):
    return ShapeDataset(args.tosca_data_path, num_feat_per_identity=1, **vars(args)), None

def get_data_shapenet_chair(args, **kwargs):
    return ShapeDataset(args.shapnet_chair_data_path, num_feat_per_identity=1, **vars(args)), None

def get_data_shapenet_chair_val(args, **kwargs):
    return ShapeDataset(args.shapnet_chair_val_data_path, num_feat_per_identity=1, **vars(args)), None

def get_data_shapenet_airplane(args, **kwargs):
    return ShapeDataset(args.shapnet_airplane_data_path, num_feat_per_identity=1, **vars(args)), None

def get_data_shapenet_airplane_val(args, **kwargs):
    return ShapeDataset(args.shapnet_airplane_val_data_path, num_feat_per_identity=1, **vars(args)), None

def get_data_polyhaven_chair(args, **kwargs):
    return ShapeDataset(args.polyhaven_chair_data_path, num_feat_per_identity=1, **vars(args)), None

def get_data_polyhaven_animals(args, **kwargs):
    return ShapeDataset(args.polyhaven_animals_data_path, num_feat_per_identity=1, **vars(args)), None

def get_data_dt4d_shape(args, kfold=None, few_shot=False,):
    if args.mask is not None:
        assert kfold is None and not few_shot, "No splitting available when passing mask"
    num_feat_per_identity = 5
    if kfold is not None:
        train_val_partition, test_partition = sample_data_set(args, args.smal_data_path, kfold, few_shot, num_feat_per_identity=num_feat_per_identity)
        return train_val_partition, test_partition, None
    return ShapeDataset(args.deforming_things_path, num_feat_per_identity=num_feat_per_identity, **vars(args)), None

def get_data_smal_ours_shape(args, kfold=None, few_shot=False, num_feat_per_identity = 10):
    if args.mask is not None:
        assert kfold is None and not few_shot, "No splitting available when passing mask"
    # num_feat_per_identity = 10 # Determined in dataset creation step
    with torch.no_grad():
        sm = SMALMesh()
        betas = torch.zeros((1, 41))
        _, _ , weights = sm.get_mesh(betas, 0) # Skinning weights are consistent across 
    
    if kfold is not None:
        train_val_partition, test_partition = sample_data_set(args, args.smal_data_path, kfold, few_shot, num_feat_per_identity=num_feat_per_identity)
        return train_val_partition, test_partition, weights.to(args.device)

    return ShapeDataset(args.smal_ours_data_path, num_feat_per_identity=num_feat_per_identity, **vars(args)), weights.to(args.device)

def get_data_smal_dummy(args, kfold=None, few_shot=False):
    return get_data_smal_ours_shape(args, kfold, few_shot, 1)

class AMASS(Dataset):
    def __init__(self, 
                 data_path, 
                 device='cuda', 
                 include_hands=False, 
                 seq_len=20,
                 stride=1,
                 filter_string=None):
        super(Dataset).__init__()
        self._data_path = data_path
        self._device = device
        self._include_hands = include_hands
        self.seq_len = seq_len
        self._thetas = []
        self._proc_labels = []
        self._latents = []
        self._buffers = []
        self.stride = stride
        self.filter_string = filter_string
        self.seq_len = seq_len

        print("Loading data...")
        data = joblib.load(self._data_path)
        self._process_data(data, max_len=self.seq_len, stride=stride)
        del data
        print("Data loaded and processed.")

    def _calc_len(self, data):
        for i in range(len(self._data)):
            print(data[1])
    
    def _process_data(self, data, max_len=20, stride=1):
        if self._include_hands:
            end = data['pose_alls'][0].shape[1]
        else:
            end = 66
        
        for i in range(len(data['pose_alls'])):
            current_pose = data['pose_alls'][i][:,:end]
            current_label = data['text_proc_labels'][i]

            if self.filter_string is not None:
                if not self.filter_string in current_label[0]:
                    continue

            current_pose = current_pose[::stride][:max_len]
            current_label = current_label[::stride][:max_len]

            if len(current_pose) < max_len:
                buffer = max_len - len(current_pose)
                current_pose = np.pad(current_pose, ((0, buffer), (0, 0)), mode='edge')
                current_label = np.pad(current_label, (0, buffer), mode='edge')
            else:
                buffer = max_len

            self._thetas.append(current_pose)
            self._proc_labels.append(current_label)
            self._buffers.append(buffer)

        self._thetas = np.stack(self._thetas, axis=0)
        self._proc_labels = np.stack(self._proc_labels, axis=0)
        self._buffers = np.array(self._buffers)

        self._thetas = torch.tensor(self._thetas, device=self._device, dtype=torch.float32)
        self._buffers = torch.tensor(self._buffers, device=self._device, dtype=torch.int32)

        print(f"Processed {len(self._thetas)} sequences.")

    
    def __len__(self):
        return len(self._thetas)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self._thetas[idx], self._proc_labels[idx], self._buffers[idx]

class ShapeDataset(Dataset):
    @staticmethod
    def check_data_leakage(d1, d2):
        for d1_path in d1.obj_paths:
            assert d1_path not in d2.obj_paths, f"Data leakage: {d1_path} present in both datasets."
        print("No data leakage detected.")
    
    @staticmethod
    def num_identites(path):
        folders = os.listdir(path)
        folders = [os.path.join(path, folder) for folder in folders if os.path.isdir(os.path.join(path, folder))]
        return len(folders)
    
    @staticmethod
    def bb_norm(vertices):
        center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
        vertices -= center
        vertices /= np.abs(vertices).max()
        return vertices
    
    @staticmethod
    def merge_into(d1, d2):
        d1.feature_noise_p = max(d1.feature_noise_p, d2.feature_noise_p)
        d1.fps_p = max(d1.fps_p, d2.fps_p)
        d1.sampling_ratio = max(d1.sampling_ratio, d2.sampling_ratio)

        d1_has_betas = d1.betas != None
        d2_has_betas = d2.betas != None
        use_betas = d1_has_betas or d2_has_betas

        if use_betas and d1_has_betas:
            betas = d1.betas
        else:
            betas = [None] * len(d1)

        d1_has_thetas = d1.thetas != None
        d2_has_thetas = d2.thetas != None
        use_thetas = d1_has_thetas or d2_has_thetas
        if use_thetas and d1_has_thetas:
            thetas = d1.thetas
        else:    
            thetas = [None] * len(d1)

        d1_is_dt4d = d1.is_dt4d
        d2_is_dt4d = d2.is_dt4d
        use_dt4d = d1_is_dt4d or d2_is_dt4d
        d1.is_dt4d = use_dt4d
        if use_dt4d and d1_is_dt4d:
            vmasks = []
            sequences = []
        else:
            vmasks = [None] * len(d1)
            sequences = [None] * len(d1)

        for i in range(len(d2)):
            d1.feat_paths.append(d2.feat_paths[i])
            d1.obj_paths.append(d2.obj_paths[i])
            if use_betas:
                if d2_has_betas:
                    betas.append(d2.betas[i])
                else:
                    betas.append(None)
            if use_thetas:
                if d2_has_thetas:
                    thetas.append(d2.thetas[i])
                else:
                    thetas.append(None)
            if use_dt4d:
                if d2_is_dt4d:
                    vmasks.append(d2.vmasks[i])
                    sequences.append(d2.sequences[i])

        if use_betas:
            d1.betas = betas
        if use_dt4d:
            d1.vmasks = vmasks
            d1.sequences = sequences
        d1.folders = d1.folders + d2.folders 
        d1.solvers = [None] * len(d1)
        d1.distances = [None] * len(d1)
        d1.sampled_indices = [None] * len(d1)
        d1.counters = [d1.reset_counter] * len(d1)
        d1.transforms = d1.transforms + d2.transforms

        return d1

    def __init__(self, path, num_feat_per_identity = 5, feature_noise_p=0, fps_p=30, sampling_ratio=1, device='cuda', mask=None, norm=True, transform=None, *args, **kwargs):
        super().__init__()
        self.path = path
        self.num_feat_per_identity = num_feat_per_identity
        self.device = device
        self.feature_noise_p = feature_noise_p
        self.fps_p = fps_p
        self.sampling_ratio = sampling_ratio
        self.mask = mask
        self.norm = norm

        self.folders = os.listdir(self.path)
        self.betas = None
        if 'betas.pt' in self.folders:
            self.betas = torch.load(os.path.join(self.path, 'betas.pt'), map_location=device)
            if self.betas.ndim == 3:
                self.betas = self.betas.view(torch.mul(*self.betas.shape[:2]), -1)
            self.betas = [beta[None] for beta in self.betas]
        self.thetas = None
        if 'thetas.pt' in self.folders:
            self.thetas = torch.load(os.path.join(self.path, 'thetas.pt'), map_location=device)
            if self.thetas.ndim == 3:
                self.thetas = self.thetas.view(torch.mul(*self.thetas.shape[:2]), -1)
            self.thetas = [theta[None] for theta in self.thetas]
        # Remove any non-folders
        self.folders = [os.path.join(self.path, folder) for folder in self.folders if os.path.isdir(os.path.join(self.path, folder))]
        self.folders = sorted_nicely(self.folders)
        self.feat_paths = []
        self.obj_paths = []

        self.is_dt4d = 'DeformingThings' in self.path
        if self.is_dt4d:
            self.source_folder_dt4d = kwargs['source_folder_dt4d']
        self.is_bt3d = 'bt3d' in self.path
        self.vmasks = []
        self.sequences = []
        self.transforms = []

        for folder_path in self.folders:
            if self.is_dt4d:
                vmask = np.load(os.path.join(folder_path, "v_mask0.npy"))
                sequences_path = os.path.join(folder_path, 'setup.txt')
                with open(sequences_path, 'r') as f:
                    sequences = f.readlines()[1:]
                sequences = [seq.split(',')[1] for seq in sequences]

            for i in range(self.num_feat_per_identity):
                feat_path = os.path.join(folder_path, f"features_{i}.pt")
                obj_path = os.path.join(folder_path, f"{i}.obj")

                if not os.path.exists(feat_path): raise Exception(f"{feat_path} missing.")
                if not os.path.exists(obj_path): raise Exception(f"{obj_path} missing.")

                self.feat_paths.append(feat_path)
                self.obj_paths.append(obj_path)
                self.transforms.append(transform)
                
                if self.is_dt4d:
                    self.vmasks.append(vmask)
                    self.sequences.append(sequences[i])

        self.solvers = [None] * len(self)
        self.reset_counter = 10
        self.counters = [self.reset_counter] * len(self)
        self.distances = [None] * len(self)
        self.sampled_indices = [None] * len(self)
        self.do_bb_norm = True
        self.force_resample = True

    def deferred_masking(self, mask):
        if mask is not None:
            assert np.max(mask) < len(self.feat_paths), "Highest mask index is bigger than number of samples"
            if self.betas is not None:
                self.betas = [self.betas[i] for i in mask]
            if self.thetas is not None:
                self.thetas = [self.thetas[i] for i in mask]
            self.feat_paths = [self.feat_paths[i] for i in mask]
            self.obj_paths = [self.obj_paths[i] for i in mask]
            self.transforms = [self.transforms[i] for i in mask]

            if self.is_dt4d:
                self.vmasks = [self.vmasks[i] for i in mask]
                self.sequences = [self.sequences[i] for i in mask]
            # self.folders = [self.folders[i] for i in mask] # TODO: Check wether this still works

        self.solvers = [None] * len(self)
        self.counters = [self.reset_counter] * len(self)
        self.distances = [None] * len(self)
        self.sampled_indices = [None] * len(self)
        
    def get_smal_shape_family_id(self, i):
        return int(self.obj_paths[i].split(os.sep)[-2])
    
    def get_animation(self, i):
        assert self.is_dt4d
        vmask = self.vmasks[i]
        seq = self.sequences[i]

        seq_path = os.path.join(self.source_folder_dt4d, seq, seq + ".anime") 
        _, _, _, vertices, _, offset_data = anime_read(seq_path)
        offset_data = offset_data[:,vmask]
        v_tgt = vertices[vmask]
        v_tgt = np.repeat(v_tgt[None], len(offset_data), 0) + offset_data

        return v_tgt, seq

    def __len__(self):
        return len(self.feat_paths)

    def __getitem__(self, i):
        if type(i) == np.ndarray or type(i) == np.array:
            i = i[0]

        if type(i) == torch.tensor:
            i = i.item()

        # Load obj
        mesh = trimesh.load_mesh(self.obj_paths[i], process=False, ignore_materials=True)
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

        if self.do_bb_norm:
            vertices = ShapeDataset.bb_norm(vertices)
        
        # Load features
        features = torch.load(self.feat_paths[i], map_location=self.device)
        features = features.to(torch.float32)
        if self.feature_noise_p > 0:
            features = features + torch.randn_like(features) * features.std() * self.feature_noise_p
        features = features / features.norm(dim=-1, keepdim=True)

        # Get geodesics
        sampled_indicies, distances = None, None
        if self.fps_p > 0:
            
            if self.force_resample or self.counters[i] == self.reset_counter:
                solver = self.solvers[i]
                sampled_indicies, distances, solver = FPS_potpourri(vertices, faces, p=self.fps_p, solver=solver, rnd=True)
                sampled_indicies = torch.tensor(sampled_indicies, device=self.device)
                distances = torch.tensor(distances, device=self.device)
                if self.solvers[i] is None: self.solvers[i] = solver
                if not self.force_resample:
                    self.sampled_indices[i] = sampled_indicies
                    self.distances[i] = distances
                    self.counters[i] = 0
            else:
                distances = self.distances[i]
                sampled_indicies = self.sampled_indices[i]
                self.counters[i] += 1
            if self.norm:
                distances = distances / distances.max()
            

        vertices = torch.tensor(vertices, device=self.device, dtype=torch.float32)
        faces = torch.tensor(faces, device=self.device)
        # Subsample jacobians and features, etc
        if self.sampling_ratio < 1:
            raise NotImplementedError("Subsampling not implemented.")
            mask = torch.randperm(faces.shape[0])[:int(self.sampling_ratio * faces.shape[0])]
            sampled_indicies = torch.arange(0, len(sampled_indicies)) + len(mask)
            mask = torch.cat([mask, sampled_indicies])

        betas = None
        thetas = None
        if self.betas is not None:
            betas = self.betas[i]
        if self.thetas is not None:
            thetas = self.thetas[i]

        if self.transforms[i] is not None:
            vertices = self.transforms[i] @ vertices.T
            vertices = vertices.T
        return vertices, faces, features, sampled_indicies, distances, betas, thetas


DATA = {
    'smal' : get_data_smal_shape,
    'shrec19': get_data_shrec19_shape,
    'shrec20': get_data_shrec20_shape,
    'surreal': get_data_surreal_shape,
    'dt4d': get_data_dt4d_shape,
    'smpl': get_data_smpl_shape,
    'smal_ours': get_data_smal_ours_shape,
    'smal_dummy': get_data_smal_dummy,
    'shapenet': get_data_shapenet,
    'tosca': get_data_tosca,
    'shapenet_chair': get_data_shapenet_chair,
    'shapenet_chair_val': get_data_shapenet_chair_val,
    'shapenet_airplane': get_data_shapenet_airplane,
    'shapenet_airplane_val': get_data_shapenet_airplane_val,
    'polyhaven_chair': get_data_polyhaven_chair,
    'polyhaven_animals': get_data_polyhaven_animals,
}
