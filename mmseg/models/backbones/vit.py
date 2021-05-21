import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, lecun_normal_
from functools import partial
from collections import OrderedDict
import math
from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger
from ..builder import BACKBONES
from mmcv.cnn import build_norm_layer
import copy


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_grid = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patch_grid[0] * self.patch_grid[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        # assert (H,W) in [(512, 512), (1024, 256), (1536, 384), (2048, 512), (2560, 640),(3072, 768),(3584, 896)] \
        #     or (W,H) in [(512, 512), (1024, 256), (1536, 384), (2048, 512), (2560, 640),(3072, 768),(3584, 896)], \
        #     f"Wrong input image size ({H}*{W})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


@BACKBONES.register_module()
class VIT(nn.Module):
    """ VisionTransformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.depth = depth
        self.img_size = img_size
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # self.stage_0_0 = nn.Sequential(
        #     nn.Conv2d(384, 384, 3, 1, 1),
        #     build_norm_layer(dict(type='BN', requires_grad=True), 384)[1],
        #     nn.ReLU(),
        # )
        # self.stage_0_1 = nn.Sequential(
        #     nn.Conv2d(384, 384, 3, 1, 1),
        #     build_norm_layer(dict(type='BN', requires_grad=True), 384)[1],
        #     nn.ReLU(),
        # )
        # self.stage_1 = nn.Sequential(
        #     nn.Conv2d(384, 384, 3, 1, 1),
        #     build_norm_layer(dict(type='BN', requires_grad=True), 384)[1],
        #     nn.ReLU(),
        # )
        # self.stage_3 = nn.Conv2d(384, 384, 3, 2, 1)

    def init_weights(self, pretrained=None):    
        def _init_weights(m, n: str = '', head_bias: float = 0., jax_impl: bool = False):
            """ ViT weight initialization
            * When called without n, head_bias, jax_impl args it will behave exactly the same
            as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
            * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
            """
            if isinstance(m, nn.Linear):
                if n.startswith('head'):
                    nn.init.zeros_(m.weight)
                    nn.init.constant_(m.bias, head_bias)
                elif n.startswith('pre_logits'):
                    lecun_normal_(m.weight)
                    nn.init.zeros_(m.bias)
                else:
                    if jax_impl:
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            if 'mlp' in n:
                                nn.init.normal_(m.bias, std=1e-6)
                            else:
                                nn.init.zeros_(m.bias)
                    else:
                        trunc_normal_(m.weight, std=.02)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
            elif jax_impl and isinstance(m, nn.Conv2d):
                # NOTE conv was left to pytorch default in my original init
                lecun_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
  
        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward_features(self, x):
        # x: [bs, channel, h, w]
        _, __, h, w = x.shape
        # use when test
        if h != self.img_size or w != self.img_size:
            tmp_pos_embed = self.pos_embed
            tmp_pos_embed = tmp_pos_embed.reshape(-1, 32, 32, self.embed_dim).permute(0, 3, 1, 2)
            tmp_pos_embed = torch.nn.functional.interpolate(
                tmp_pos_embed, size=(int(h/16), int(w/16)), mode='bicubic', align_corners=False)
            tmp_pos_embed = tmp_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
        else:
            tmp_pos_embed = self.pos_embed
        x = self.patch_embed(x)    
        x = self.pos_drop(x + tmp_pos_embed)
        outs = []
        for i in range(self.depth):
            x = self.blocks[i](x)
            x = self.norm(x)
            if i+1 in [2,4,10,12]:
                # hw = int(self.patch_embed.num_patches ** 0.5)
                ph = int(h/16)
                pw = int(w/16)
                out = x.view(-1, ph, pw, self.embed_dim).permute(0, 3, 1, 2).contiguous()
                # deconvolution
                # if i+1 == 2:
                #     out = self.stage_0_0(out)
                #     out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
                #     out = self.stage_0_1(out)
                #     out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
                # elif i + 1 == 4:
                #     out = self.stage_1(out)
                #     out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
                # elif i + 1 == 12:
                #     out = self.stage_3(out)
                outs.append(out)
        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

