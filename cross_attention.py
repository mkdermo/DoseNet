import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
# from vit.gcn import GraphConvolution

def pair(t):
    return t if isinstance(t, tuple) else (t, t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_qv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_k = nn.Linear(dim, inner_dim * 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, x1, attn_mat=None, dots_para=None, mat_para=None):    
        qqvv = self.to_qv(x).chunk(2, dim = -1)   # qqvv[0].size()=[1, 4096, 192], qqvv[1].size()=[1, 4096, 192]
        q, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qqvv) # q = v = [1, 6, 4096, 32]
        kk = self.to_k(x1).chunk(1, dim = -1)   # kk[0].size()=[1, 4096, 192]
        k = rearrange(kk[0], 'b n (h d) -> b h n d', h = self.heads)    # k = [1, 6, 4096, 32]

        out = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # dots = [1, 6, 4096, 4096] or [1, 6, 512, 512]
        if attn_mat is not None:  
             # attn_mat.size()=[1, h, 4096, 4096] or [1, h, 512, 512]
            out = out * dots_para + attn_mat * mat_para

        out = self.attend(out)   # attn = [1, 6, 4096, 4096] or [1, 6, 512, 512]
        out = self.dropout(out)

        out = torch.matmul(out, v)  # out.size = [1, 6, 4096, 32]
        out = rearrange(out, 'b h n d -> b n (h d)')  # out.size = [1, 4096, 192]
        return self.to_out(out)


class Cross_Attention_Module(nn.Module):
    def __init__(self, image_size, patch_size, channels, dim, mlp_dim, heads = 8,\
                 dim_head = 64, depth=12, emb_dropout = 0, dropout = 0., dist_mat_fc=0, \
                 img_gcns=0, relative_position_encoding=False, two_ca=False):
        super().__init__()
        self.relative_position_encoding = relative_position_encoding
        self.img_gcns = img_gcns
        self.two_ca = two_ca
        self.depth = depth
        image_height, image_width, image_depth = pair(image_size)
        patch_height, patch_width, patch_depth = pair(patch_size)
        assert image_height % patch_height == 0 and \
               image_width % patch_width == 0 and image_depth % patch_depth == 0
    
        num_patches = (image_height // patch_height) * \
                      (image_width // patch_width) * (image_depth // patch_depth)
        patch_dim = channels * patch_height * patch_width * patch_depth

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=patch_height, \
                       p2 = patch_width, p3 = patch_depth),
            nn.Linear(patch_dim, dim),
        )
        self.img_pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.rad_pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.img_dropout = nn.Dropout(emb_dropout)
        self.rad_dropout = nn.Dropout(emb_dropout)

        if self.img_gcns > 0 or self.relative_position_encoding:
            self.dist_fc1 = nn.Linear(3, 1, bias=True)
            # self.dist_fc2 = nn.Linear(dist_mat_fc, 1, bias=True)

        # config norm, crs_attn, ff
        self.img_norm, self.rad_norm, self.mlp_norm, self.crs_attn, self.ff, self.gcn_list = \
            nn.ModuleList([]),nn.ModuleList([]),nn.ModuleList([]),nn.ModuleList([]),\
            nn.ModuleList([]), nn.ModuleList([])
        self.dots_paras, self.mat_paras = [], []

        for i in range(depth):
            self.img_norm.append(nn.LayerNorm(dim))
            self.rad_norm.append(nn.LayerNorm(dim))
            self.mlp_norm.append(nn.LayerNorm(dim))  
            self.crs_attn.append(Cross_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
            self.ff.append(FeedForward(dim, mlp_dim, dropout = dropout))

            # if self.img_gcns > 0:
            #     l = nn.ModuleList([])
            #     for _ in range(self.img_gcns):
            #         l.append(GraphConvolution(in_features=dim, out_features=dim))
            #     self.gcn_list.append(l)

            if self.relative_position_encoding:
                self.dots_paras.append(nn.Parameter(torch.randn(1)).cuda())
                self.mat_paras.append(nn.Parameter(torch.randn(1)).cuda())

        self.fuse = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
            Rearrange('b (h w d) (p1 p2 p3 c) -> b c (h p1) (w p2) (d p3)', 
                      p1 = patch_height, p2 = patch_width, p3 = patch_depth, 
                      h = image_height // patch_height, w = image_width // patch_width, d = image_depth // patch_depth)
        )

    def forward(self, img, rad, msk_mat=None, dist_mat=None):   
        img = self.to_patch_embedding(img)   # img.size = [1, 4096, 384]
        rad = self.to_patch_embedding(rad)   # rad.size = [1, 4096, 384]
        img += self.img_pos_embedding  # self.img_pos_embedding.size = [1, 4096, 384]
        rad += self.rad_pos_embedding  # self.rad_pos_embedding.size = [1, 4096, 384]

        img = self.img_dropout(img)   # img.size = [1, 4096, 384]
        rad = self.rad_dropout(rad) 

        if dist_mat is not None and msk_mat is not None:  
            # msk_mat = self.xyz_mat_length_3_to_heads(dist_mat) * msk_mat # (bz, r, r, 3) -> # [bz, n_head, r, r]
            dist_mat_gcn = 1 / dist_mat
            dist_mat_gcn = self.xyz_mat_length_3_to_1(dist_mat_gcn) * msk_mat # (bz, r, r, 3) -> # (bz, r, r, 1)
            # print( msk_mat.size(), dist_mat_gcn.size()) # [1, 4096, 4096], [1, 4096, 4096]
        elif dist_mat is not None and self.img_gcns > 0:  
            dist_mat_gcn = 1 / dist_mat
            dist_mat_gcn = self.xyz_mat_length_3_to_1(dist_mat_gcn)

        for i in range(self.depth):
            # graph neural network
            for j in range(self.img_gcns):
                img = self.gcn_list[i][j].forward(img, dist_mat_gcn) + img
                self.check_inf_nan(img)
            # crs_attn 
            if not self.relative_position_encoding:
                if self.two_ca:
                    img = self.crs_attn[i].forward(self.img_norm[i](img), self.rad_norm[i](rad)) +\
                        self.crs_attn[i].forward(self.rad_norm[i](rad), self.img_norm[i](img)) + img # img.size = [1, 4096, 384]
                else:
                    img = self.crs_attn[i].forward(self.img_norm[i](img), self.rad_norm[i](rad)) + img
            else:
                img = self.crs_attn[i].forward(self.img_norm[i](img), self.rad_norm[i](rad), \
                    dist_mat_gcn, self.dots_paras[i],self.mat_paras[i]) + img 
            img = self.ff[i](self.mlp_norm[i](img))    # img.size = [1, 4096, 384]

        img = self.fuse(img) 
        return img

    def check_inf_nan(self, img):
        a, b = torch.isinf(img), torch.isnan(img)
        if True in a: print('inf inf inf inf inf inf inf')
        if True in b: print('nan nan nan nan nan nan nan')

    def xyz_mat_length_3_to_1(self, xyz_mat):
        '''input: [bz, r, r, 3]
           output: [bz, r, r, 1]'''
        flatten_xyz_mat = xyz_mat.view(-1, 3).to(torch.float32)    # [bz*r*r, 3]
        box_size_per_head = list(xyz_mat.shape[:3]) + [1]          # box_size_per_head=[bz, r, r, 1]
        flatten_xyz_mat = self.dist_fc1(flatten_xyz_mat)           # [bz*r*r, fc_len]
        # flatten_xyz_mat = self.dist_fc2(flatten_xyz_mat)           # [bz*r*r, 1]
        flatten_xyz_mat = flatten_xyz_mat.view(box_size_per_head)  # [bz, r, r, 1]
        return torch.squeeze(flatten_xyz_mat, -1)                  # [bz, r, r]
