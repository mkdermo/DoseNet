import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode  

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        
        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        r = nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode) 
        return r


def resize3D(input, dim1, dim2, dim3, is_msk=False, use_cuda=False):
    d1 = torch.linspace(-1, 1, dim1)
    d2 = torch.linspace(-1, 1, dim2)
    d3 = torch.linspace(-1, 1, dim3)
    meshx, meshy, meshz = torch.meshgrid(d1, d2, d3, indexing='ij')
    if use_cuda: grid = torch.stack((meshz, meshy, meshx), -1).cuda()#.cuda(int(use_cuda))
    else:        grid = torch.stack((meshz, meshy, meshx), -1)
    grid = grid.unsqueeze(0)
    grid = grid.expand((input.size(0), *grid.size()[1:]))
    if is_msk:
        out = F.grid_sample(input, grid, align_corners=True, mode='nearest')
    else:
        out = F.grid_sample(input, grid, align_corners=True, mode='bilinear')

    if use_cuda: return out, grid
    else:        return out


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class ConvBlock_No_Bn(nn.Module):
    def __init__(self, ndims, in_channels, out_channels, stride=1, kernel_size=3, padding=1):
        super().__init__()
        if kernel_size == 1:
            padding = 0
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, activation=True):
        out = self.main(x)
        if activation:
            out = self.activation(out)
        return out