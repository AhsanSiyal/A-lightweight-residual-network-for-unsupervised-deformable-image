# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:53:16 2022

@author: Ahsan
"""

import torch.nn as nn
import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as nnf
from arch.util_g1 import (
    ResidualConv,
    ASPP,
    AttentionBlock,
    Upsample_,
    Squeeze_Excite_Block,
)


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

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class ResUnet(nn.Module):
    def __init__(self, shape, channel, filters=[8, 16, 32, 32, 64]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(filters[0]),
            nn.ReLU(),
            nn.Conv3d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv3d(channel, filters[0], kernel_size=3, padding=1)
        )


        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)


        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)


        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        self.output_layer = nn.Sequential(nn.Conv3d(filters[0], 3, 1))
        
        self.transformer = SpatialTransformer(shape)

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        # x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x1)

        # x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x2)

        # x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x3)

        x5 = self.aspp_bridge(x4)

        # x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x5)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        # x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x6)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        # x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x7)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        flow = self.output_layer(x9)
        
        moving = x[:, 0:1, :, :]
        y = self.transformer(moving, flow)

        return y, flow
    
#%%

# path= "C:/Drive/Workspace/data_sets/normalized_selected/atlas/20.nii"
# device=torch.device('cuda:0')
# vol_size=(160,192,224)
# enc_f1 = (16, 32, 32, 32)
# dec_f1 = (32, 32, 16, 16)
# atlas_vol = nib.load(path).get_fdata()
# atlas_vol = np.expand_dims(atlas_vol, 0)
# atlas_vol = np.expand_dims(atlas_vol, 0)
# net = ResUnet(vol_size,2)
# net.cuda(device)
# cat = np.concatenate([atlas_vol, atlas_vol], axis=1)
# #print(net)
# atlas_tensor = torch.tensor(cat, device=device, dtype=torch.float)
# con = net(atlas_tensor)
# print(con[0].shape)
    