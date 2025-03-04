import numpy as np
import torch
import torch.nn as nn
from model.cross_attention import Cross_Attention_Module
from model.layers import SpatialTransformer, resize3D, ConvBlock, ConvBlock_No_Bn
from model.losses import MSE, Dice, Grad


class Dosenet(nn.Module):

    def __init__(self, cf, inshape=(160,160,160), infeats=1, nb_features=None, ct_features=None, seg_loss_gamma=0.1, \
                 use_mask=True, pool_len=5, smooth_reg=0.02):
        super().__init__()
        crs_attn_modules = nn.ModuleList([])
        for i, n in enumerate(nb_features[0]):
            crs_attn = Cross_Attention_Module(image_size=cf.img_dims[i], patch_size=cf.patch_sizes[i], \
                channels=cf.channels[i], depth=cf.depths[i], dim=cf.dims[i], heads=cf.heads[i], mlp_dim=cf.mlp_dims[i],\
                dim_head=cf.dim_heads[i], dropout=cf.dropouts[i], emb_dropout=cf.emb_dropouts[i])
            crs_attn_modules.append(crs_attn)
        
        self.backbone = Dosenet_Module(inshape=inshape, infeats=infeats, nb_features=nb_features, \
            crs_attn_modules=crs_attn_modules, parotid_resized_shape=cf.parotid_resized_shape)
        self.rad_ct_dvf_layer = ConvBlock_No_Bn(ndims=3, in_channels=nb_features[-1][-1], out_channels=3, stride=1)        
        self.last_layer = ConvBlock_No_Bn(ndims=3, in_channels=3, out_channels=3, stride=1, kernel_size=1, padding=0)        

        self.smooth_reg = smooth_reg
        self.seg_loss_gamma = seg_loss_gamma
        self.spatial_transf_img = SpatialTransformer(size=inshape) #name='transformer')
        self.spatial_transf_msk = SpatialTransformer(size=inshape, mode='nearest')
        self.use_mask = use_mask

    def forward(self, rad_img_l, rad_img_r, pre_rt_img, post_rt_img, \
                pre_rt_msk_l=None, pre_rt_msk_r=None, post_rt_msk=None):

        self.post_rt_img, self.post_rt_msk = post_rt_img, post_rt_msk
        pre_rt_img, rad_img_l, rad_img_r = pre_rt_img.cuda(), rad_img_l.cuda(), rad_img_r.cuda()

        # left parotid dvf
        disp_tensor_l = self.backbone(pre_rt_img, rad_img_l, pre_rt_msk_l.clone())
        disp_tensor_l = self.rad_ct_dvf_layer.forward(disp_tensor_l, activation=False)
        disp_tensor_l = disp_tensor_l * pre_rt_msk_l.cuda()
        # right parotid dvf
        disp_tensor_r = self.backbone(pre_rt_img, rad_img_r, pre_rt_msk_r.clone())
        disp_tensor_r = self.rad_ct_dvf_layer.forward(disp_tensor_r, activation=False)
        disp_tensor_r = disp_tensor_r * pre_rt_msk_r.cuda()
        # no parotid dvf
        mid_msk = torch.ones_like(pre_rt_msk_r)
        mid_msk[pre_rt_msk_r == 1] = 0
        mid_msk[pre_rt_msk_l == 1] = 0
        disp_tensor_mid = (disp_tensor_l + disp_tensor_r) * mid_msk.cuda() / 2
        # sum total dvf
        self.disp_tensor = disp_tensor_mid + disp_tensor_l + disp_tensor_r
        self.disp_tensor = self.last_layer.forward(self.disp_tensor, activation=False)
        # spatial transform
        pre_rt_msk = pre_rt_msk_r + pre_rt_msk_l
        self.moved_img_tensor = self.spatial_transf_img(pre_rt_img, self.disp_tensor)
        self.moved_msk_tensor = self.spatial_transf_msk(pre_rt_msk.cuda(), self.disp_tensor) # (bz, 1, d, d, d)

        del pre_rt_msk, pre_rt_img, rad_img_l, rad_img_r, pre_rt_msk_r, pre_rt_msk_l  # save gpu memory
        return self.disp_tensor, self.moved_img_tensor, self.moved_msk_tensor

    def loss(self):
        bz, chl, d1, d2, d3 = self.post_rt_img.size()
        self.moved_img_tensor, self.img_grid = resize3D(self.moved_img_tensor, dim1=d1, dim2=d2, dim3=d3, is_msk=False, use_cuda=True)
        self.moved_msk_tensor, self.msk_grid = resize3D(self.moved_msk_tensor, dim1=d1, dim2=d2, dim3=d3, is_msk=True, use_cuda=True)
        mse_loss = MSE.loss(self.post_rt_img.cuda(), self.moved_img_tensor)
        smooth_loss = Grad().loss(None, self.moved_img_tensor)

        self.moved_msk_tensor = self.to_one_hot(self.moved_msk_tensor)    # (bz, 2, d1, d2, d3)
        dice_loss = Dice.loss(self.post_rt_msk.cuda(), self.moved_msk_tensor)

        total_loss = mse_loss + dice_loss * self.seg_loss_gamma + smooth_loss * self.smooth_reg
        return total_loss, mse_loss.detach().item(), dice_loss.detach().item(), smooth_loss.detach().item()

    def to_one_hot(self, moved_msk_tensor):
        zero_layer = moved_msk_tensor * -1 + 1
        return torch.cat((zero_layer, moved_msk_tensor), 1)


class Dosenet_Module(nn.Module):
    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False, 
                 crs_attn_modules=None, 
                 parotid_resized_shape=None):
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res
        self.crs_attn_modules = crs_attn_modules
        self.parotid_resized_shape = parotid_resized_shape

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure dose encoder (2nd down-sampling path)
        dose_prev_nf = 1
        self.dose_encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]  # nf is conv out channel
                convs.append(ConvBlock(ndims, dose_prev_nf, nf))
                dose_prev_nf = nf
            self.dose_encoder.append(convs)       

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def find_bbox(self, prtd_msk, cls):
        '''prtd_msk.size = (n,n,n)'''
        ind = (prtd_msk==cls).nonzero(as_tuple=False).numpy()
        z_max, z_min = max(ind[:, 0]), min(ind[:, 0])
        y_max, y_min = max(ind[:, 1]), min(ind[:, 1])
        x_max, x_min = max(ind[:, 2]), min(ind[:, 2])
        return [z_max, z_min, y_max, y_min, x_max, x_min]

    def forward(self, x, dose, prtd_msks):
        x = torch.cat((x, dose), dim=1)

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            # do conv
            for conv in convs:
                x = conv(x)
            for conv in self.dose_encoder[level]: 
                dose = conv(dose)
            
            # roi cropping and resizing
            z_max, z_min, y_max, y_min, x_max, x_min = self.find_bbox(prtd_msks[0][0], cls=1)
            img_dim = self.parotid_resized_shape[level]
            attn_x, _ = resize3D(x[:, :, z_min:z_max, y_min:y_max, x_min:x_max],\
                                    dim1=img_dim, dim2=img_dim, dim3=img_dim, \
                                    is_msk=False, use_cuda=True)
            attn_dose, _ = resize3D(dose[:, :, z_min:z_max, y_min:y_max, x_min:x_max],\
                                    dim1=img_dim, dim2=img_dim, dim3=img_dim, \
                                    is_msk=False, use_cuda=True)
            # attn
            attn_x = self.crs_attn_modules[level].forward(img=attn_x, rad=attn_dose)
            attn_x, _ = resize3D(attn_x, dim1=z_max-z_min, dim2=y_max-y_min, dim3=x_max-x_min, \
                                    is_msk=False, use_cuda=True)
            x_clone = x.clone()
            x_clone[:, :, z_min:z_max, y_min:y_max, x_min:x_max] = attn_x
            x_history.append(x_clone)

            # pooling
            x = self.pooling[level](x)
            dose = self.pooling[level](dose)
            prtd_msks = self.pooling[level](prtd_msks)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x

