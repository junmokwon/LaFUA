from typing import Optional, Tuple, Union, Type, List
import warnings
from functools import lru_cache

import torch
from torch import nn, einsum
import torch.nn.functional as F


# Imported and modified from the official AnyUp repository:
# https://github.com/wimmerth/anyup
# T. Wimmer et al., "AnyUp: Universal Feature Upsampling", ICLR 2026.

# ./anyup/utils/img.py
def create_coordinate_3d(d, h, w, start=0.0, end=1.0, device=None, dtype=None):
    z = torch.linspace(start, end, d, device=device, dtype=dtype)
    y = torch.linspace(start, end, h, device=device, dtype=dtype)
    x = torch.linspace(start, end, w, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack((zz, yy, xx), -1).view(1, d * h * w, 3)


# ./anyup/layers.convolutions.py
class ResBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_groups=8,
                 pad_mode="zeros", norm_fn=None, activation_fn=nn.SiLU, use_conv_shortcut=False,
                 pad_depth=True):
        super().__init__()
        N = (lambda c: norm_fn(num_groups, c)) if norm_fn else (lambda c: nn.Identity())
        p = kernel_size // 2
        padding = p if pad_depth else [0, p, p]
        kernel_sizes = kernel_size if pad_depth else [1, kernel_size, kernel_size]
        self.block = nn.Sequential(
            N(in_channels),
            activation_fn(),
            nn.Conv3d(in_channels, out_channels, kernel_sizes, padding=padding, padding_mode=pad_mode, bias=False),
            N(out_channels),
            activation_fn(),
            nn.Conv3d(out_channels, out_channels, kernel_sizes, padding=padding, padding_mode=pad_mode, bias=False),
        )
        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False, padding_mode=pad_mode)
            if use_conv_shortcut or in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


# ./anyup/utils/gaussian_derivative_initialization.py
compute_basis_size_3d = lambda order: ((order + 1) * (order + 2) * (order + 3)) // 6


# ./anyup/utils/gaussian_derivative_initialization.py
def herme_vander_torch(x, degree):
    vander = torch.ones(x.shape + (degree + 1,), device=x.device, dtype=x.dtype)
    if degree > 0:
        vander[..., 1] = x
    for i in range(2, degree + 1):
        vander[..., i] = x * vander[..., i-1] - (i-1) * vander[..., i-2]
    return vander


# ./anyup/utils/gaussian_derivative_initialization.py
def gauss_deriv_3d(max_order, device, dtype, kernel_size, sigma=None):
    sigma = (kernel_size // 2) / 1.645 if sigma is None else sigma
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, dtype=dtype, device=device)
    z_dist = x / sigma
    g = torch.exp(-0.5 * z_dist ** 2) / (sigma * (2.0 * torch.pi) ** 0.5)
    He = herme_vander_torch(z_dist, max_order)
    
    derivs_1d = [(((-1) ** n) / (sigma ** n)) * He[:, n] * g for n in range(max_order + 1)]
    bank = []
    for o in range(max_order + 1):
        for i in range(o + 1):
            for j in range(o - i + 1):
                k = o - i - j
                K = torch.einsum('i,j,k->ijk', derivs_1d[i], derivs_1d[j], derivs_1d[k])
                bank.append(K)
    return torch.stack(bank, 0)


# ./anyup/layers/feature_unification.py
class LearnedFeatureUnification3d(nn.Module):
    def __init__(self, out_channels: int, kernel_size: int = 3, init_gaussian_derivatives: bool = False):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if init_gaussian_derivatives:
            order = 0
            while compute_basis_size_3d(order) < out_channels:
                order += 1
            basis = gauss_deriv_3d(order, 'cpu', torch.float32, kernel_size)[:out_channels, None]
            self.basis = nn.Parameter(basis)
        else:
            self.basis = nn.Parameter(torch.randn(out_channels, 1, kernel_size, kernel_size, kernel_size))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = features.shape
        x = self._depthwise_conv(features, self.basis, self.kernel_size).view(b, self.out_channels, c, d, h, w)
        attn = F.softmax(x, dim=1)
        return attn.mean(dim=2)

    @staticmethod
    def _depthwise_conv(feats, basis, k):
        b, c, d, h, w = feats.shape
        p = k // 2
        x = F.pad(feats, (p, p, p, p, p, p), value=0)
        x = F.conv3d(x, basis.repeat(c, 1, 1, 1, 1), groups=c)
        mask = torch.ones(1, 1, d, h, w, dtype=x.dtype, device=x.device)
        denom = F.conv3d(F.pad(mask, (p, p, p, p, p, p), value=0), 
                         torch.ones(1, 1, k, k, k, dtype=x.dtype, device=x.device))
        return x / denom


# ./anyup/layers/attention/attention_masking.py
def window3d(low_res: Tuple[int, int, int], high_res: Tuple[int, int, int], ratio: float, device: str = "cpu"):
    D, H, W = high_res
    Ld, Lh, Lw = low_res

    def get_bounds(res_high, res_low):
        pos = (torch.arange(res_high, device=device, dtype=torch.float32) + 0.5) / res_high
        lo = ((pos - ratio).clamp(0.0, 1.0) * res_low).floor().long()
        hi = ((pos + ratio).clamp(0.0, 1.0) * res_low).ceil().long()
        return lo, hi

    d0, d1 = get_bounds(D, Ld)
    h0, h1 = get_bounds(H, Lh)
    w0, w1 = get_bounds(W, Lw)

    grid_d0, grid_h0, grid_w0 = torch.meshgrid(d0, h0, w0, indexing="ij")
    grid_d1, grid_h1, grid_w1 = torch.meshgrid(d1, h1, w1, indexing="ij")

    return torch.stack([grid_d0, grid_d1, grid_h0, grid_h1, grid_w0, grid_w1], dim=-1)


# ./anyup/layers/attention/attention_masking.py
@lru_cache
def compute_attention_mask_3d(high_res, low_res, window_size_ratio, device="cpu"):
    D, H, W = high_res
    D_, H_, W_ = low_res
    windows = window3d(low_res, high_res, window_size_ratio, device)
    q = D * H * W

    d0, d1 = windows[..., 0].reshape(q, 1), windows[..., 1].reshape(q, 1)
    h0, h1 = windows[..., 2].reshape(q, 1), windows[..., 3].reshape(q, 1)
    w0, w1 = windows[..., 4].reshape(q, 1), windows[..., 5].reshape(q, 1)

    id_ = torch.arange(D_, device=device)
    ih_ = torch.arange(H_, device=device)
    iw_ = torch.arange(W_, device=device)

    d_ok = (id_ >= d0) & (id_ < d1)
    h_ok = (ih_ >= h0) & (ih_ < h1)
    w_ok = (iw_ >= w0) & (iw_ < w1)

    mask = d_ok.unsqueeze(-1).unsqueeze(-1) & h_ok.unsqueeze(1).unsqueeze(-1) & w_ok.unsqueeze(1).unsqueeze(1)
    return ~mask.reshape(q, D_ * H_ * W_)


# ./anyup/layers/attention/chunked_attention.py
class CrossAttention(nn.Module):
    def __init__(self, qk_dim, num_heads,
                 q_chunk_size: Optional[int] = None,
                 store_attn: bool = False):
        super().__init__()
        self.norm_q = nn.RMSNorm(qk_dim)
        self.norm_k = nn.RMSNorm(qk_dim)
        self.q_chunk_size = q_chunk_size
        self.store_attn = store_attn
        self.attention = nn.MultiheadAttention(
            embed_dim=qk_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

    @torch.no_grad()
    def _slice_mask(self, mask, start, end):
        if mask is None:
            return None
        if mask.dim() == 2:
            return mask[start:end, :]
        elif mask.dim() == 3:
            return mask[:, start:end, :]
        else:
            raise ValueError("attn_mask must be 2D or 3D")

    def forward(self, query, key, value, mask=None,
                q_chunk_size: Optional[int] = None,
                store_attn: Optional[bool] = None):
        q_chunk_size = self.q_chunk_size if q_chunk_size is None else q_chunk_size
        store_attn = self.store_attn if store_attn is None else store_attn

        val = key

        query = self.norm_q(query)
        key = self.norm_k(key)

        if q_chunk_size is None or query.size(1) <= q_chunk_size:
            _, attn = self.attention(query, key, val,
                                     average_attn_weights=True,
                                     attn_mask=mask)
            features = einsum("b i j, b j d -> b i d", attn, value)
            return features, (attn if store_attn else None)

        B, Q, _ = query.shape
        outputs = []
        attns = [] if store_attn else None

        for start in range(0, Q, q_chunk_size):
            end = min(start + q_chunk_size, Q)
            q_chunk = query[:, start:end, :]
            mask_chunk = self._slice_mask(mask, start, end)

            _, attn_chunk = self.attention(q_chunk, key, val,
                                           average_attn_weights=True,
                                           attn_mask=mask_chunk)
            out_chunk = einsum("b i j, b j d -> b i d", attn_chunk, value)
            outputs.append(out_chunk)
            if store_attn:
                attns.append(attn_chunk)

        features = torch.cat(outputs, dim=1)
        attn_scores = torch.cat(attns, dim=1) if store_attn else None
        return features, attn_scores


# ./anyup/layers/attention/chunked_attention.py
class CrossAttentionBlock3d(nn.Module):
    def __init__(self, qk_dim, num_heads, window_ratio: float = 0.1, q_chunk_size: Optional[int] = None):
        super().__init__()
        self.cross_attn = CrossAttention(qk_dim, num_heads, q_chunk_size=q_chunk_size)
        self.window_ratio = window_ratio
        self.conv3d = nn.Conv3d(qk_dim, qk_dim, kernel_size=3, padding=1, bias=False)

    def forward(self, q, k, v, q_chunk_size: Optional[int] = None):
        q = self.conv3d(q)
        if self.window_ratio > 0:
            attn_mask = compute_attention_mask_3d(q.shape[-3:], k.shape[-3:], self.window_ratio, q.device)
        else:
            attn_mask = None
        
        b, c, d, h, w = q.shape
        _, _, d_k, h_k, w_k = k.shape
        
        q = q.permute(0, 2, 3, 4, 1).view(b, d * h * w, -1)
        k = k.permute(0, 2, 3, 4, 1).view(b, d_k * h_k * w_k, -1)
        v = v.permute(0, 2, 3, 4, 1).reshape(b, d_k * h_k * w_k, -1)

        features, _ = self.cross_attn(q, k, v, mask=attn_mask, q_chunk_size=q_chunk_size)
        return features.view(b, d, h, w, -1).permute(0, 4, 1, 2, 3)


# ./anyup/layers/positional_encoding.py
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# ./anyup/layers/positional_encoding.py
class RoPE3d(nn.Module):
    def __init__(self, dim: int, theta: int = 100):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.freqs = nn.Parameter(torch.empty(3, self.dim))

    def _device_weight_init(self):
        part = self.dim // 6
        freqs_1d = self.theta ** torch.linspace(0, -1, part)
        freqs_3d = torch.zeros(3, self.dim)
        for i in range(3):
            freqs_3d[i, i*2*part : (i*2+1)*part] = freqs_1d
            freqs_3d[i, (i*2+1)*part : (i+1)*2*part] = freqs_1d
        self.freqs.data.copy_(freqs_3d * 2 * torch.pi)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        angle = coords @ self.freqs
        return x * angle.cos() + rotate_half(x) * angle.sin()


# ./anyup/model.py
class AnyUp3d(nn.Module):
    def __init__(self, input_dim=1, qk_dim=128, kernel_size=3, kernel_size_lfu=5, window_ratio=0.1, num_heads=4, init_gaussian_derivatives=False,
                 pad_depth=True):
        super().__init__()
        self.qk_dim = qk_dim
        self._rb_args = dict(kernel_size=3, num_groups=8, pad_mode="reflect", norm_fn=nn.GroupNorm, activation_fn=nn.SiLU, pad_depth=pad_depth)

        self.image_encoder = self._make_encoder(input_dim, kernel_size)
        self.key_encoder = self._make_encoder(qk_dim, 1)
        self.query_encoder = self._make_encoder(qk_dim, 1)
        self.key_features_encoder = self._make_encoder(None, 1, first_layer_k=kernel_size_lfu, init_gaussian_derivatives=init_gaussian_derivatives)

        self.cross_decode = CrossAttentionBlock3d(qk_dim, num_heads, window_ratio=window_ratio)
        self.aggregation = self._make_encoder(2 * qk_dim, 3, padding_mode="reflect" if pad_depth else "zeros")
        self.rope = RoPE3d(qk_dim)
        self.rope._device_weight_init()

    def _make_encoder(self, in_ch, k, layers=2, first_layer_k=0, init_gaussian_derivatives=False, padding_mode="reflect"):
        pre = (nn.Conv3d(in_ch, self.qk_dim, k, padding=k // 2, padding_mode=padding_mode, bias=False)
               if first_layer_k == 0 else
               LearnedFeatureUnification3d(self.qk_dim, first_layer_k, init_gaussian_derivatives))
        blocks = [ResBlock3d(self.qk_dim, self.qk_dim, **self._rb_args) for _ in range(layers)]
        return nn.Sequential(pre, *blocks)

    def forward(self, image, features, output_size=None):
        # image: (B, C, D, H, W), features: (B, C, D_low, H_low, W_low)
        output_size = output_size if output_size is not None else image.shape[-3:]
        enc = self.image_encoder(image)
        d, h, w = enc.shape[-3:]
        
        coords = create_coordinate_3d(d, h, w, device=enc.device, dtype=enc.dtype)
        enc = enc.permute(0, 2, 3, 4, 1).view(enc.shape[0], -1, enc.shape[1])
        enc = self.rope(enc, coords)
        enc = enc.view(enc.shape[0], d, h, w, -1).permute(0, 4, 1, 2, 3)

        q = F.interpolate(self.query_encoder(enc), size=output_size, mode='trilinear')
        k = F.interpolate(self.key_encoder(enc), size=features.shape[-3:], mode='trilinear')
        kf = self.key_features_encoder(F.normalize(features, dim=1))
        k = self.aggregation(torch.cat([k, kf], dim=1))
        
        return self.cross_decode(q, k, features)


# ./anyup/loss.py
class Cosine_MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.cosine_loss = torch.nn.CosineEmbeddingLoss()

    def forward(self, pred, target):
        if target.ndim == 5:
            pred = pred.permute(0, 2, 3, 4, 1).reshape(-1, pred.shape[1])
            target = target.permute(0, 2, 3, 4, 1).reshape(-1, target.shape[1])
        elif target.ndim == 4:
            pred = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
            target = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])
        else:
            raise RuntimeError(f"Incompatible dimension: {target.ndim}")
       
        gt = torch.ones_like(target[:, 0])

        min_val = torch.min(target, dim=1, keepdim=True).values
        max_val = torch.max(target, dim=1, keepdim=True).values
        pred_normalized = (pred - min_val) / (max_val - min_val + 1e-6)
        target_normalized = (target - min_val) / (max_val - min_val + 1e-6)

        loss = self.cosine_loss(pred, target, gt) + self.mse_loss(pred_normalized, target_normalized)
        return loss


def imagenet_preprocess_3d(x, means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225), eps=1e-6):
    B, C, D, H, W = x.shape

    # Batch-wise and channel-wise minmax normalization
    x_reshaped = x.view(B, C, -1)
    v_min = x_reshaped.min(dim=2, keepdim=True)[0].view(B, C, 1, 1, 1)
    v_max = x_reshaped.max(dim=2, keepdim=True)[0].view(B, C, 1, 1, 1)
    x = (x - v_min) / torch.clamp(v_max - v_min, min=eps)

    # (B, C, D, H, W) -> (B, D, C, H, W) -> (B*D, C, H, W)
    x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)

    # Adjust input channel
    if C == 1:
        x = x.repeat(1, 3, 1, 1)
    elif C == 2:
        x = torch.cat([x, x[:, -1:, :, :]], dim=1)
    elif C > 3:
        x = x[:, :3, :, :]

    if isinstance(means, torch.Tensor):
        x = (x - means) / stds
    else:
        for c, mean, std in enumerate(zip(means, stds)):
            x[:, c, :, :] = (x[:, c, : :] - mean) / std
    return x

def imagenet_preprocess_2d(x, means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225), eps=1e-6):
    B, C, H, W = x.shape
    x_flat = x.view(B, -1)
    
    v_min = x_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    v_max = x_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    x = (x - v_min) / torch.clamp(v_max - v_min, min=eps)
    
    if C == 1:
        x = x.expand(-1, 3, -1, -1)
    elif C == 2:
        x = torch.cat([x, x[:, -1:, :, :]], dim=1)
    elif C > 3:
        x = x[:, :3, :, :]
    
    if isinstance(means, torch.Tensor):
        x = (x - means) / stds
    else:
        for c, mean, std in enumerate(zip(means, stds)):
            x[:, c, :, :] = (x[:, c, : :] - mean) / std
    return x

def imagenet_preprocess(x, means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225), eps=1e-6):
    if x.ndim == 5:
        return imagenet_preprocess_3d(x, means=means, stds=stds, eps=eps)
    elif x.ndim == 4:
        return imagenet_preprocess_2d(x, means=means, stds=stds, eps=eps)
    else:
        raise RuntimeError(f"Incompatible dimension: {x.ndim}")


from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder


class UNetDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None
                 ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs


        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output


from torch.nn.modules.conv import _ConvNd
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0


def restore_from_backbone(x: torch.Tensor, B: int, D: int):
    BD, C, h, w = x.shape
    return x.view(B, D, C, h, w).transpose(1, 2)


class ResidualEncoderUNet(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        backbone_dim: int,
        pad_depth: bool,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
        stem_channels: int = None,
        window_ratio: float = 0.1,
    ):
        super().__init__()

        self.key_to_encoder = "encoder.stages"
        self.key_to_stem = "encoder.stem"
        self.keys_to_in_proj = ("encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0")

        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, (
            "n_blocks_per_stage must have as many entries as we have "
            f"resolution stages. here: {n_stages}. "
            f"n_blocks_per_stage: {n_blocks_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            "n_conv_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {n_stages} "
            f"stages, so it should have {n_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )
        self.encoder = ResidualEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            block,
            bottleneck_channels,
            return_skips=True,
            disable_default_stem=False,
            stem_channels=stem_channels,
        )
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)
        self.anyup = AnyUp3d(input_dim=input_channels, pad_depth=pad_depth, window_ratio=window_ratio)
        self.backbone_proj = nn.Conv3d(in_channels=features_per_stage[-1], out_channels=backbone_dim, kernel_size=1)
        out_proj = []
        for s in range(-2, -len(features_per_stage)-1, -1):
            out_proj.append(
                nn.Conv3d(in_channels=features_per_stage[-1], out_channels=features_per_stage[s], kernel_size=1)
            )
        self.out_proj = nn.ModuleList(out_proj)

    def forward(self, x):
        orig_x = x
        encoder = self.encoder
        decoder = self.decoder
        deep_supervision = decoder.deep_supervision

        skips = encoder(x)
        pred_lr_feats = skips[-1]
        lres_input = skips[-1]
        seg_outputs = []
        n_stages = len(decoder.stages)
        for s in range(n_stages):
            x = decoder.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = decoder.stages[s](x)
            if s == (n_stages - 1):
                pred_hr_feats = self.anyup(orig_x, self.out_proj[s](pred_lr_feats), orig_x.shape[-3:])
                x = x + pred_hr_feats
                seg_outputs.append(decoder.seg_layers[-1](x))
            else:
                downsampled_x = F.interpolate(orig_x, scale_factor=1/(2**(n_stages - s - 1)), mode='trilinear', 
                                              align_corners=False, antialias=False)
                pred_hr_feats = self.anyup(downsampled_x, self.out_proj[s](pred_lr_feats), downsampled_x.shape[-3:])
                x = x + pred_hr_feats
                del downsampled_x
                if deep_supervision:
                    seg_outputs.append(decoder.seg_layers[s](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not deep_supervision:
            return seg_outputs[0]
        else:
            return seg_outputs, pred_lr_feats

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size
        )

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


import os
import pydoc
import functools
import math
import numpy as np
from torch import autocast

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class LaFUA_Trainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.backbone = self.build_backbone().to(self.device)

        self.num_epochs = 250
        self.initial_lr = 1e-3
        self.weight_decay = 1e-5

        self.backbone_patch_size = 16
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.num_crops = 4
        self.cosine_mse_loss = Cosine_MSE()
        self.lambda_lfa = 1.5

    @staticmethod
    def build_backbone() -> nn.Module:
        raise NotImplementedError

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])
        if enable_deep_supervision is not None:
            architecture_kwargs['deep_supervision'] = enable_deep_supervision
        network = ResidualEncoderUNet(
            backbone_dim=384,
            pad_depth=False,
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            **architecture_kwargs
        )
        return network

    def set_deep_supervision_enabled(self, enabled: bool):
        if enabled:
            super().set_deep_supervision_enabled(enabled)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(),
                                      lr=self.initial_lr,
                                      weight_decay=self.weight_decay,
                                      eps=1e-4,  # 1e-8 might cause nans in fp16
                                      betas=(0.9, 0.999))
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, scheduler

    def _batch_multi_crop_3d(self, batch, n_stages, num_crops=4):
        if n_stages <= 4:
            img_lr = F.interpolate(batch["data"], scale_factor=2 ** (5 - n_stages), mode='trilinear', 
                                   align_corners=False, antialias=False)
            img_hr = F.interpolate(batch["data"], scale_factor=2 ** (6 - n_stages), mode='trilinear', 
                                   align_corners=False, antialias=False)
        elif n_stages == 5:
            img_lr = batch["data"]
            img_hr = F.interpolate(img_lr, scale_factor=2, mode='trilinear', 
                                   align_corners=False, antialias=False)
        elif n_stages == 6:
            img_hr = batch["data"] # (B, C, D, H, W)
            img_lr = F.interpolate(img_hr, scale_factor=0.5, mode='trilinear', 
                                   align_corners=False, antialias=False)
        else:
            raise RuntimeError("_batch_multi_crop_3d only supports stage 5 and 6.")

        crops = []
        coords = []
        patch_size = self.backbone_patch_size
        B, C, D, H, W = img_hr.shape
        d, h, w = img_lr.shape[2:]

        max_depth = max(0, D - d)
        max_top = max(0, H - h)
        max_left = max(0, W - w)

        for b in range(B):
            for _ in range(num_crops):
                z = patch_size * torch.randint(0, (max_depth // patch_size) + 1, (1,)).item()
                y = patch_size * torch.randint(0, (max_top // patch_size) + 1, (1,)).item()
                x = patch_size * torch.randint(0, (max_left // patch_size) + 1, (1,)).item()
                
                crop = img_hr[b, :, z:z+d, y:y+h, x:x+w]
                crops.append(crop)
                coords.append((z // patch_size, y // patch_size, x // patch_size))
        hr = torch.stack(crops, dim=0) # (B*K, C, d, h, w)

        means = self.imagenet_mean
        stds = self.imagenet_std
        batch["hr_image"] = imagenet_preprocess(hr, means=means, stds=stds)
        batch["lr_image"] = imagenet_preprocess(img_lr, means=means, stds=stds)
        batch["guidance_crop"] = torch.tensor(coords, dtype=torch.long)
        batch["upsampling_size"] = (D // patch_size, H // patch_size, W // patch_size)
        return batch

    def train_step(self, batch: dict) -> dict:
        num_crops = self.num_crops
        n_stages = self.configuration_manager.network_arch_init_kwargs["n_stages"]
        batch = self._batch_multi_crop_3d(batch, n_stages, num_crops=num_crops)
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        hr_image = batch['hr_image'].to(self.device, non_blocking=True)
        lr_image = batch['lr_image'].to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            with torch.no_grad():
                backbone = functools.partial(self.backbone.get_intermediate_layers, n=1, reshape=True, norm=True)
                if n_stages >= 5:
                    div = data.shape[2] // (2 ** (n_stages - 5))
                else:
                    div = data.shape[2] * (2 ** (5 - n_stages))

                hr_feats = backbone(hr_image)[0]
                hr_feats = restore_from_backbone(hr_feats, data.shape[0] * num_crops, div)

                lr_feats = backbone(lr_image)[0]
                lr_feats = restore_from_backbone(lr_feats, data.shape[0], div)
            output, pred_lr_feats = self.network(data)

            target_size = batch["upsampling_size"]
            anyup_hr_feats = self.network.anyup(data, lr_feats, target_size)
            d, h, w = hr_feats.shape[-3:]
            p = self.backbone_patch_size
            d //= p

            pred_lr_feats = self.network.backbone_proj(pred_lr_feats)
            with torch.no_grad():
                lr_feats = F.interpolate(lr_feats, scale_factor=(1/p, 1.0, 1.0), mode='trilinear', 
                                         align_corners=False, antialias=False)
            l = self.loss(output, target) + \
                self.lambda_lfa * self.cosine_mse_loss(pred_lr_feats, lr_feats)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output, _ = self.network(data)
            del data
            l = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def perform_actual_validation(self, save_probabilities: bool = False):
        pass


class LaFUA_Trainer_DINOv3_ViTS16(LaFUA_Trainer):
    @staticmethod
    def build_backbone():
        backbone = torch.hub.load(
            os.environ["LFA_Backbone_Repo"],
            'dinov3_vits16',
            source='local',
            weights=os.environ["LFA_Backbone_Checkpoint"]
        ).eval()
        for p in backbone.parameters():
            p.requires_grad = False
        return backbone

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])
        if enable_deep_supervision is not None:
            architecture_kwargs['deep_supervision'] = enable_deep_supervision
        network = ResidualEncoderUNet(
            backbone_dim=384,
            pad_depth=False,
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            window_ratio=0.1,
            **architecture_kwargs
        )
        return network


class LaFUA_Trainer_DINOv3_ViTS16_PadDepth(LaFUA_Trainer_DINOv3_ViTS16):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        architecture_kwargs = dict(**arch_init_kwargs)
        for ri in arch_init_kwargs_req_import:
            if architecture_kwargs[ri] is not None:
                architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])
        if enable_deep_supervision is not None:
            architecture_kwargs['deep_supervision'] = enable_deep_supervision
        network = ResidualEncoderUNet(
            backbone_dim=384,
            pad_depth=True,
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            window_ratio=0.1,
            **architecture_kwargs
        )
        return network

