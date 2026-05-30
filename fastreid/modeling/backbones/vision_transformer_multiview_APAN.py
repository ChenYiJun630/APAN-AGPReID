""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929
The official jax code is released and available at https://github.com/google-research/vision_transformer
Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2020 Ross Wightman
"""

import logging
import math
import pdb
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastreid.layers import DropPath, trunc_normal_, to_2tuple
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY
from functools import reduce
from operator import mul
import copy
from fastreid.modeling.backbones.wtconv import WTConv2d
from fastreid.modeling.backbones.resnet import ResNet,Bottleneck


logger = logging.getLogger(__name__)

class GatedSubtractiveMLP(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        self.dim = dim
        hidden_dim = hidden_dim or dim * 2

        # MLP for residual transform
        self.residual_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

        # Gate network: input is [meta || view], output in [0,1]^dim
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, meta, view):
        # Ensure input shape match
        assert meta.shape == view.shape, "meta and view tokens must have the same shape"
        # Compute gate: based on both meta and view
        gate_input = torch.cat([meta, view], dim=-1)  # shape [B, 2D]
        gate = self.gate_net(gate_input)              # shape [B, D], in (0,1)

        # Compute residual (nonlinear transformation of difference)
        diff = meta - view                            # shape [B, D]
        residual = self.residual_mlp(diff)            # shape [B, D]

        # Apply gate to residual and subtract from meta
        updated_meta = meta - gate * residual         # shape [B, D]

        return updated_meta

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
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):

    def __init__(
        self,
        embedding_dim: int,         # 输入channel
        num_heads: int,             # attention的head数
        downsample_rate: int = 1,   # 下采样
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        # qkv获取
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    #头部分离
    def _separate_heads(self, x, num_heads: int) :
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head
    #头重组
    def _recombine_heads(self, x):
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q, k, v) :
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        # B,N_heads,N_tokens,C_per_head
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B,N_heads,N_tokens,C_per_head
        # Scale
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        # Get output
        out = attn @ v
        # # B,N_tokens,C
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out

class OutAttenBlock(nn.Module):
    def __init__(self, dim, num_heads, 
                 mlp_ratio=4.0,
                 activation = nn.ReLU,   
                 attention_downsample_rate: int = 1, 
                 norm_layer=nn.LayerNorm):
        
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.norm3 = norm_layer(dim)

    def forward(self, q, k, v):
        attn_out = self.cross_attn(q, k, v)
        queries = q + attn_out
        queries = self.norm1(queries)
        queries = queries + self.attn(queries)
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        return queries


    
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


class PrototypeRelationGeneration(nn.Module):
    def __init__(self, node_dim, num_candidate_prototypes, k, num_fixed_prototypes=0,
                 num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_prototype = num_candidate_prototypes 
        self.k = k
        self.num_fixed_prototypes = num_fixed_prototypes
        self.total_prototypes  = num_fixed_prototypes + num_candidate_prototypes  
        self.head_dim = node_dim // num_heads
        self.prototype_base = nn.Parameter(torch.Tensor(self.total_prototypes , node_dim))
        nn.init.xavier_uniform_(self.prototype_base)
        self.context_net = nn.Linear(2 * node_dim, self.total_prototypes  * node_dim)
        self.pre_head_proj = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, X):
        B, N, D = X.shape

        patch_tokens = X[:, 1:]  # [B, N-1, D]
        weights = patch_tokens.mean(dim=-1, keepdim=True).softmax(dim=1)
        attn_pooled = (weights * patch_tokens).sum(dim=1)  # [B,D] 
        cls_token = X[:, 0]
        context_cat = torch.cat([cls_token, attn_pooled], dim=-1)

        prototype_offsets = self.context_net(context_cat).view(B, self.total_prototypes , D)
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets  

        
        X_proj = self.pre_head_proj(X)
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, self.total_prototypes , self.num_heads, self.head_dim).permute(0, 2, 1, 3)

       
        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.total_prototypes , self.head_dim).transpose(1, 2)

        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling
        logits = logits.view(B, self.num_heads, N, self.total_prototypes ).mean(dim=1)
        logits = self.dropout(logits)
        A = logits
    
        global_scores = A.sum(dim=1) #[B,48]
        dynamic_scores = global_scores[:, self.num_fixed_prototypes:] 
       
        topk_vals, topk_indices = torch.topk(dynamic_scores, self.k, dim=-1)  
        topk_indices += self.num_fixed_prototypes

        mask = torch.zeros_like(global_scores)  
        mask[:, :self.num_fixed_prototypes] = 1.0  
        
        mask.scatter_(dim=1, index=topk_indices, src=torch.ones_like(topk_vals))
        mask = mask.unsqueeze(1) 

        A = A * mask
        A = F.softmax(A, dim=1)

        self.last_mask = mask.detach()
        self.last_A = A.detach()
        return A


class PrototypesAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, delta1=1.0, delta2=1.0):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # log-prior strength
        self.delta_token2prototype = delta1
        self.delta_prototype2token = delta2

        self.dropout = nn.Dropout(dropout)

        # ------------------------------------------------------------------
        # Token -> Prototype
        # prototype query comes from prototype_init = A^T X
        # token key/value comes from X
        # ------------------------------------------------------------------
        self.token_norm_t2p = nn.LayerNorm(dim)
        self.prototype_norm_t2p = nn.LayerNorm(dim)

        self.token2prototype_q = nn.Linear(dim, dim)
        self.token2prototype_k = nn.Linear(dim, dim)
        self.token2prototype_v = nn.Linear(dim, dim)
        self.token2prototype_out = nn.Linear(dim, dim)

        # ------------------------------------------------------------------
        # Prototype -> Token
        # token query comes from X
        # prototype key/value comes from updated prototypes
        # ------------------------------------------------------------------
        self.token_norm_p2t = nn.LayerNorm(dim)
        self.prototype_norm_p2t = nn.LayerNorm(dim)

        self.prototype2token_q = nn.Linear(dim, dim)
        self.prototype2token_k = nn.Linear(dim, dim)
        self.prototype2token_v = nn.Linear(dim, dim)
        self.prototype2token_out = nn.Linear(dim, dim)

    def _split_heads(self, x):
        batch_size, seq_len, dim = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def forward(self, X, A):
        """
        Args:
            X: token features, shape [B, N, D]
            A: token-prototype assignment / prior matrix, shape [B, N, M]

        Returns:
            message: prototype-enhanced token message, shape [B, N, D]
        """

        B, N, D = X.shape
        M = A.shape[-1]

        # ==============================================================
        # 1. Token -> Prototype
        # prototype_init = A^T X
        # ==============================================================

        prototype_init = torch.bmm(A.transpose(1, 2), X)  

        tokens_t2p = self.token_norm_t2p(X)
        prototypes_t2p = self.prototype_norm_t2p(prototype_init)

        prototype_query = self._split_heads(self.token2prototype_q(prototypes_t2p))  
        token_key = self._split_heads(self.token2prototype_k(tokens_t2p))            
        token_value = self._split_heads(self.token2prototype_v(tokens_t2p))          

        token2prototype_logits = torch.matmul(prototype_query, token_key.transpose(-2, -1)) * self.scale  

        token2prototype_prior = A.transpose(1, 2).unsqueeze(1)  
        token2prototype_logits = token2prototype_logits + self.delta_token2prototype * torch.log(token2prototype_prior.clamp_min(1e-6))

        token2prototype_attn = torch.softmax(token2prototype_logits, dim=-1)  
        token2prototype_attn = self.dropout(token2prototype_attn)

        prototypes = torch.matmul(token2prototype_attn, token_value)  
        prototypes = self._merge_heads(prototypes)                    
        prototypes = self.dropout(self.token2prototype_out(prototypes))  

        # ==============================================================
        # 2. Prototype -> Token
        # ==============================================================

        tokens_p2t = self.token_norm_p2t(X)
        prototypes_p2t = self.prototype_norm_p2t(prototypes)

        token_query = self._split_heads(self.prototype2token_q(tokens_p2t))         
        prototype_key = self._split_heads(self.prototype2token_k(prototypes_p2t))    
        prototype_value = self._split_heads(self.prototype2token_v(prototypes_p2t))  

        prototype2token_logits = torch.matmul(token_query, prototype_key.transpose(-2, -1)) * self.scale  

        prototype2token_prior = A.unsqueeze(1)  
        prototype2token_logits = prototype2token_logits + self.delta_prototype2token * torch.log(prototype2token_prior.clamp_min(1e-6))

        prototype2token_attn = torch.softmax(prototype2token_logits, dim=-1) 
        prototype2token_attn = self.dropout(prototype2token_attn)

        message = torch.matmul(prototype2token_attn, prototype_value)  
        message = self._merge_heads(message)                           
        message = self.dropout(self.prototype2token_out(message))       

        return message

class APAN_Network(nn.Module):
    def __init__(self, embed_dim, num_candidate_prototypes=64, top_k=16, num_fixed_prototypes=8,
                 num_heads=8, dropout=0.1,delta1=1.0, delta2=1.0,use_gate=True):
        super().__init__()
        self.protoRelation = PrototypeRelationGeneration(
            node_dim=embed_dim,
            num_candidate_prototypes=num_candidate_prototypes,
            k=top_k,
            num_fixed_prototypes=num_fixed_prototypes,
            num_heads=num_heads,
            dropout=dropout
        )
        self.use_gate = use_gate
        self.edge_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.GELU(),)
        self.node_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim),nn.GELU(),)
        self.proto_attn = PrototypesAttention(embed_dim, num_heads=num_heads, dropout=dropout,delta1=delta1,delta2 =delta2)
        if self.use_gate:
            self.gate = nn.Parameter(torch.tensor(1.0))
 

    def forward(self, X):
   
        A = self.protoRelation(X)                 
        message = self.proto_attn(X, A)       

        if self.use_gate:
            return X + self.gate * message
        
        return X + message




#原始图片，不重叠
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

#输入的数据为CNN得到的特征，而不是图片
class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

#将输入图像分割成具有重叠区域的Patch（原始图片）
class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """

    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)  # [64, 8, 768]
        return x


class VisionTransformer_multiview(nn.Module):
    """ Vision Transformer
        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
            - https://arxiv.org/abs/2010.11929
        Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
            - https://arxiv.org/abs/2012.12877
        """

    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 stride_size=16, 
                 in_chans=3, 
                 embed_dim=768,
                 depth=12, 
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None,
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 camera=0, 
                 drop_path_rate=0., 
                 hybrid_backbone=None,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                 sie_xishu=1.0, #SIE缩放系数
                 inner_sub=True, 
                 local_feat=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed_overlap(
                img_size=img_size, 
                patch_size=patch_size, 
                stride_size=stride_size, 
                in_chans=in_chans,
                embed_dim=embed_dim)

        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.view_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 2, embed_dim))

        self.cam_num = camera
        self.sie_xishu = sie_xishu
        self.local_feat = local_feat
        # Initialize SIE Embedding
        if camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i], 
                norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)
        self.inner_sub = inner_sub

        self.gated_mlp = GatedSubtractiveMLP(dim=embed_dim)




    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'view_token'}

    def forward(self, x, camera_id=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        view_tokens = self.view_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, view_tokens, x), dim=1)

        if self.cam_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)
        #特征提取块
        if self.local_feat:#若使用局部特征提取，则只遍历 Transformer 块的前 depth-1 层。
            for blk in self.blocks[:-1]:
                x = blk(x)
                # perform inner sub
                if self.inner_sub:
                    x[:, 0] = x[:, 0] - x[:, 1]
                  
            return x

        else: 
            for blk in self.blocks:
                x = blk(x)
                # perform inner sub
                if self.inner_sub:
                    x[:, 0] = x[:, 0] - x[:, 1]
            
            x = self.norm(x)
            return x[:, 0].reshape(x.shape[0], -1, 1, 1), x[:, 1].reshape(x.shape[0], -1, 1, 1)
            
    def load_param(self, pretrain_path):
        try:
            state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
            logger.info(f"Loading pretrained model from {pretrain_path}")

            if 'model' in state_dict:
                state_dict = state_dict.pop('model')
            if 'state_dict' in state_dict:
                state_dict = state_dict.pop('state_dict')
            for k, v in state_dict.items():
                if 'head' in k or 'dist' in k:
                    continue
                if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                    # For old models that I trained prior to conv based patchification
                    O, I, H, W = self.patch_embed.proj.weight.shape
                    v = v.reshape(O, -1, H, W)
                elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                    # To resize pos embedding when using model at different size from pretrained weights
                    if 'distilled' in pretrain_path:
                        logger.info("distill need to choose right cls token in the pth.")
                        v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                    v = resize_pos_embed(v, self.pos_embed.data, self.patch_embed.num_y, self.patch_embed.num_x, 2)
                state_dict[k] = v
        except FileNotFoundError as e:
            logger.info(f'{pretrain_path} is not found! Please check this path.')
            raise e
        except KeyError as e:
            logger.info("State dict keys error! Please check the state dict.")
            raise e

        incompatible = self.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class build_resnet(nn.Module):
    def __init__(self,cfg):
        super(build_resnet, self).__init__()
        last_stride = 1
        model_path = "/workspace/ReID/AGPReID/pretrain_model/resnet50-19c8e357.pth"
        self.mode = 1#1
        pretrain_choice = "imagenet"
        self.neck = "bnneck"
        self.neck_feat = "before"

        self.in_planes = 2048
        self.base = ResNet(last_stride=last_stride,block=Bottleneck,layers=[3, 4, 6, 3])


        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            # print('Loading pretrained ImageNet model......from {}'.format(model_path))
        self.gap = GeM()
        # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        # self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, cam_label=None, view_label=None, label=None):  # label is unused if self.cos_layer == 'no'
        mid_fea = self.base(x)
        global_feat = self.gap(mid_fea)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        
        return mid_fea, global_feat
        

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

class LocalRefinementUnits(nn.Module):
    def __init__(self, dim, out_dim=768, kernel=1, choice=True):
        super().__init__()
        self.LRU = choice
        self.channels = dim
        self.out_dim = out_dim
        self.dwconv = nn.Conv2d(self.channels, self.channels, kernel, 1, padding=0, groups=self.channels)
        #深度可分离卷积，只对每个通道独立卷积不做跨通道），保留局部空间信息，不增加通道之间的信息交互
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.ptconv = nn.Conv2d(self.channels, self.out_dim, 1, 1)
        #逐点卷积（Pointwise Convolution），1x1 卷积，用于通道转换（通常结合深度卷积使用）将通道数变为 out_dim。
        self.bn2 = nn.BatchNorm2d(self.out_dim)
        self.act1 = nn.PReLU()#使用了 PReLU（带学习参数的ReLU） 激活函数，比普通 ReLU 更灵活
        self.act2 = nn.PReLU()
        self.act = nn.ReLU()

    def forward(self, x):
        if self.LRU:
            x = self.act1(self.bn1(self.dwconv(x)))
            x = self.act2(self.bn2(self.ptconv(x)))
        else:
            x = self.act2(self.bn2(self.ptconv(x)))
        return x
    
class MultiScaleLRU(nn.Module):
    def __init__(self, dim, out_dim=768):
        super().__init__()
        self.conv3x3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv5x5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.bn = nn.BatchNorm2d(dim*2)
        self.ptconv = nn.Conv2d(dim*2, out_dim, 1)
        self.act = nn.PReLU()

    def forward(self, x):
        x1 = self.conv3x3(x)
        x2 = self.conv5x5(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.act(self.bn(x))
        x = self.ptconv(x)
        return x


#位置嵌入调整函数
#保证了嵌入特征在不同分辨率下依然具有相似的空间布局。
def resize_pos_embed(posemb, posemb_new, hight, width, cls_token_num):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224

    ntok_new = posemb_new.shape[1]#获取新的嵌入数量

    posemb_token, posemb_grid = posemb[:, :cls_token_num], posemb[0, 1:]
    #提取分类标记 (CLS token) 的嵌入，形状为 (1, cls_token_num, embed_dim)。
    #posemb_grid：提取其余 patch 的嵌入，形状为 (num_patches, embed_dim)
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))#如果原始 patch 数为 196（14x14），则 gs_old = 14。

    logger.info('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape,
                                                                                                      posemb_new.shape,
                                                                                                      hight,
                                                                                                      width))
    #位置嵌入调整（插值）
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)

    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb

class Vision_Transformer_APAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        input_size = cfg.INPUT.SIZE_TRAIN
        pretrain = cfg.MODEL.BACKBONE.PRETRAIN
        pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
        depth = cfg.MODEL.BACKBONE.DEPTH
        sie_xishu = cfg.MODEL.BACKBONE.SIE_COE
        stride_size = cfg.MODEL.BACKBONE.STRIDE_SIZE
        drop_ratio = cfg.MODEL.BACKBONE.DROP_RATIO
        drop_path_ratio = cfg.MODEL.BACKBONE.DROP_PATH_RATIO
        attn_drop_rate = cfg.MODEL.BACKBONE.ATT_DROP_RATE
        inner_sub = cfg.MODEL.BACKBONE.INNER_SUB
        self.in_planes = 768
        self.prompt_len = cfg.MODEL.BACKBONE.PROMPT_LEN#64
        self.prompt_trans_depth = cfg.MODEL.BACKBONE.PROMPT_DEPTH
        # self.use_prompt = cfg.MODEL.BACKBONE.USE_PROMPT
        pretrain = cfg.MODEL.BACKBONE.PRETRAIN
        pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
        self.use_prm = cfg.MODEL.BACKBONE.USE_PRM
        # fmt: on

        self.use_gate = cfg.MODEL.BACKBONE.USE_GATE
        self.topk = cfg.MODEL.BACKBONE.TOPK
        self.delta1 = cfg.MODEL.BACKBONE.DELTA1
        self.delta2 = cfg.MODEL.BACKBONE.DELTA2
        
        # VDT init
        num_depth = {'small': 8, 'base': 12,}[depth]
        num_heads = {'small': 8, 'base': 12,}[depth]
        mlp_ratio = {'small': 3., 'base': 4,}[depth]
        qkv_bias = {'small': False, 'base': True}[depth]
        qk_scale = {'small': 768 ** -0.5, 'base': None,}[depth]
        self.base = VisionTransformer_multiview(
            img_size=input_size, 
            sie_xishu=sie_xishu, 
            stride_size=stride_size,
            depth=num_depth,
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale,
            drop_path_rate=drop_path_ratio, 
            drop_rate=drop_ratio,
            attn_drop_rate=attn_drop_rate, 
            inner_sub=inner_sub, local_feat=True
        )

        if pretrain:
            self.base.load_param(pretrain_path)

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )


        self.apan_network = APAN_Network(
            embed_dim=self.in_planes,
            num_candidate_prototypes=32,
            top_k=self.topk,
            num_fixed_prototypes=16,
            num_heads=8,
            dropout=0.1,
            delta1=self.delta1,
            delta2=self.delta2,
            use_gate=self.use_gate
        )



        self.norm = nn.LayerNorm(self.in_planes, eps=1e-6)
 


    def forward(self, x, camera_id=None):
        B = x.shape[0]
        # VDT
        local_features = self.base(x, camera_id=camera_id)
        local_feat = self.b1(local_features)
        global_features = local_feat[:, 0:1]#全局特征clstorch.Size([16, 1, 768])
        view_features = local_feat[:, 1:2]#视觉特征torch.Size([16, 1, 768])
        local_feat = local_features[:, 2:]#剩余特征[128,210,768]

        #Global_ALL
        global_features = global_features - view_features
        # global_features=self.HTMG(global_features,conv_feats)


        concat_token = torch.cat((global_features,local_feat),dim=1)
        concat_token = self.b1(concat_token) 
        proto_token = self.apan_network(concat_token)
        out_feat = proto_token[:, 0,:]



        return  global_features.reshape(x.shape[0], -1, 1, 1), out_feat.reshape(x.shape[0], -1, 1, 1), view_features.reshape(x.shape[0], -1, 1, 1)


@BACKBONE_REGISTRY.register()
def build_multiview_vit_backbone_APAN(cfg):
    """
    Create a Vision Transformer instance from config.
    Returns:
        SwinTransformer: a :class:`SwinTransformer` instance.
    """
    # fmt: off

    # fmt: on

    model = Vision_Transformer_APAN(cfg)
    return model
