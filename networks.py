import torch

class Block(torch.nn.Module):
    def __init__(self, input_dim, output_dim, skip=False):
        super().__init__()

        self.skip = skip

        self.fc1 = torch.nn.Linear(input_dim, input_dim)
        self.fc2 = torch.nn.Linear(input_dim, output_dim)
        self.silu1 = torch.nn.SiLU()
        self.silu2 = torch.nn.SiLU()
        self.norm1 = torch.nn.LayerNorm(input_dim)
        self.norm2 = torch.nn.LayerNorm(output_dim)

        self.block1 = torch.nn.Sequential(*[
            self.fc1, self.silu1,
        ])
        self.block2 = torch.nn.Sequential(*[
            self.fc2, self.silu2, self.norm2 
        ])
            

    def forward(self, x):
        out = self.block1(x)
        if self.skip:
            x = x + out
        else:
            x = out
        x = self.norm1(x)

        return self.block2(x)

class GenericNetwork(torch.nn.Module):
    def __init__(self, layer_nums):
        super().__init__()
        self.layer_nums = list(layer_nums)
        self.input_dim = layer_nums[0]
        self.output_dim = self.layer_nums[-1]

        layers = []
        for i in range(len(self.layer_nums)-1):
            layers.append(Block(self.layer_nums[i], self.layer_nums[i+1]))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        assert x.ndim == 3, "Input must be batch x points x features"
        return self.net(x)
    
class FeatureNetwork(torch.nn.Module):
    def __init__(self, feature_dim=2048, num_layers=4, **kwargs) -> None:
        super().__init__()

        self.feat_dim = feature_dim
        self.layers = [feature_dim // (2**n) for n in range(num_layers + 1)]
        self.encoder = GenericNetwork(self.layers) 
        self.decoder = GenericNetwork(self.layers[::-1])

    def encode(self, F, norm=True):
        f = self.encoder(F[:,None,:])
        if norm:
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
        return f

    def forward(self, F, norm=True):
        f = self.encode(F, norm=norm)
        F_hat = self.decoder(f)
        F_hat = F_hat / F_hat.norm(dim=-1, keepdim=True)
        return f[:,0], F_hat[:,0]
