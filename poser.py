from human_body_prior.body_model.body_model import BodyModel
import torch
from scripts import paths

class Poser(object):
    def __init__(self,
                 smplh_amass_model_path=paths.smplh_amass_model_path, 
                 num_betas=0,
                 batch_size=8,
                 device="cuda") -> None:
        super().__init__()
        self.smplh_amass_model_path = smplh_amass_model_path
        self.num_betas = num_betas
        self.batch_size = batch_size
        self.device = device
        self.body_model = BodyModel(self.smplh_amass_model_path, 'smplh', num_betas=self.num_betas, batch_size=batch_size).to(self.device)

    def _pose(self, pose_body, root_orient=None, betas=None):
        assert pose_body.ndim == 2, "Batch x pose"
        outs = []
        _betas = betas[None].expand(self.batch_size, -1) if betas is not None else None

        res = pose_body.shape[0] % self.batch_size
        if res != 0:
            buff = self.batch_size - res
            pose_body = torch.cat([
                pose_body, 
                torch.zeros(buff, pose_body.shape[-1]).to(pose_body.device)], 
            axis=0)
            if root_orient is not None:
                root_orient = torch.cat([
                    root_orient, 
                    torch.zeros(buff, 3).to(pose_body.device)],
                axis=0)

        for i in range(0, pose_body.shape[0], self.batch_size):
            batch = pose_body[i:i+self.batch_size]
            if root_orient is not None:
                root_batch = root_orient[i:i+self.batch_size]
            else:
                root_batch = None
            vertices = self.body_model(root_orient=root_batch, pose_body=batch, betas=_betas).vertices
            outs.append(vertices)

        outs = torch.cat(outs, axis=0)

        if res != 0:
            outs = outs[:-buff]

        return outs
        

    def pose(self, thetas, betas=None, use_orient=False, grad=False):
        assert thetas.ndim == 3, "Batch x time x pose"
        if betas is not None:
            assert betas.ndim == 1, "betas"

        b, t, p = thetas.shape
        thetas = thetas.view(b*t, p)

        root, pose_body = thetas[:, :3], thetas[:, 3:]
        if not use_orient:
            root = None

        if not grad:
            with torch.no_grad():
                vertices = self._pose(pose_body, root, betas)
        else:
            vertices = self._pose(pose_body, root, betas)

        vertices = vertices.view(b, t, -1, 3)
        return vertices