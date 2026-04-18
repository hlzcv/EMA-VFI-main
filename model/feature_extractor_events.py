import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0]*window_size[1], C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    nwB, N, C = windows.shape
    windows = windows.view(-1, window_size[0], window_size[1], C)
    B = int(nwB / (H * W / window_size[0] / window_size[1]))
    x = windows.view(
        B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def pad_if_needed(x, size, window_size):
    n, h, w, c = size
    pad_h = math.ceil(h / window_size[0]) * window_size[0] - h
    pad_w = math.ceil(w / window_size[1]) * window_size[1] - w
    if pad_h > 0 or pad_w > 0:  # center-pad the feature on H and W axes
        img_mask = torch.zeros((1, h+pad_h, w+pad_w, 1))  # 1 H W 1
        h_slices = (
            slice(0, pad_h//2),
            slice(pad_h//2, h+pad_h//2),
            slice(h+pad_h//2, None),
        )
        w_slices = (
            slice(0, pad_w//2),
            slice(pad_w//2, w+pad_w//2),
            slice(w+pad_w//2, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, window_size
        )  # nW, window_size*window_size, 1
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)
        ).masked_fill(attn_mask == 0, float(0.0))
        return nn.functional.pad(
            x,
            (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        ), attn_mask
    return x, None


def depad_if_needed(x, size, window_size):
    n, h, w, c = size
    pad_h = math.ceil(h / window_size[0]) * window_size[0] - h
    pad_w = math.ceil(w / window_size[1]) * window_size[1] - w
    if pad_h > 0 or pad_w > 0:  # remove the center-padding on feature
        return x[:, pad_h // 2 : pad_h // 2 + h, pad_w // 2 : pad_w // 2 + w, :].contiguous()
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        

class InterFrameAttention(nn.Module):
    def __init__(self, dim, motion_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.motion_dim = motion_dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.cor_embed = nn.Linear(2, motion_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.motion_proj = nn.Linear(motion_dim, motion_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2, cor, H, W, mask=None):
        B, N, C = x1.shape
        B, N, C_c = cor.shape
        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        cor_embed_ = self.cor_embed(cor)
        cor_embed = cor_embed_.reshape(B, N, self.num_heads, self.motion_dim // self.num_heads).permute(0, 2, 1, 3)
        k, v = kv[0], kv[1]    
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            nW = mask.shape[0] # mask: nW, N, N
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        c_reverse = (attn @ cor_embed).transpose(1, 2).reshape(B, N, -1)
        motion = self.motion_proj(c_reverse-cor_embed_)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, motion


class MotionFormerBlock(nn.Module):
    def __init__(self, dim, motion_dim, num_heads, window_size=0, shift_size=0, mlp_ratio=4., bidirectional=True, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super().__init__()
        self.window_size = window_size
        if not isinstance(self.window_size, (tuple, list)):
            self.window_size = to_2tuple(window_size)
        self.shift_size = shift_size
        if not isinstance(self.shift_size, (tuple, list)):
            self.shift_size = to_2tuple(shift_size)
        self.bidirectional = bidirectional
        self.norm1 = norm_layer(dim)
        self.attn = InterFrameAttention(
            dim,
            motion_dim, 
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, cor, H, W, B):
        x = x.view(2*B, H, W, -1)
        x_pad, mask = pad_if_needed(x, x.size(), self.window_size)
        cor_pad, _ = pad_if_needed(cor, cor.size(), self.window_size)

        if self.shift_size[0] or self.shift_size[1]:
            _, H_p, W_p, C = x_pad.shape
            x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            cor_pad = torch.roll(cor_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            
            if hasattr(self, 'HW') and self.HW.item() == H_p * W_p: 
                shift_mask = self.attn_mask
            else:
                shift_mask = torch.zeros((1, H_p, W_p, 1))  # 1 H W 1
                h_slices = (slice(0, -self.window_size[0]),
                            slice(-self.window_size[0], -self.shift_size[0]),
                            slice(-self.shift_size[0], None))
                w_slices = (slice(0, -self.window_size[1]),
                            slice(-self.window_size[1], -self.shift_size[1]),
                            slice(-self.shift_size[1], None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        shift_mask[:, h, w, :] = cnt
                        cnt += 1

                mask_windows = window_partition(shift_mask, self.window_size).squeeze(-1)  
                shift_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                shift_mask = shift_mask.masked_fill(shift_mask != 0, 
                                float(-100.0)).masked_fill(shift_mask == 0, 
                                float(0.0))
                                
                if mask is not None:
                    shift_mask = shift_mask.masked_fill(mask != 0, 
                                float(-100.0))
                self.register_buffer("attn_mask", shift_mask)
                self.register_buffer("HW", torch.Tensor([H_p*W_p]))
        else: 
            shift_mask = mask
        
        if shift_mask is not None:
            shift_mask = shift_mask.to(x_pad.device)
            

        _, Hw, Ww, C = x_pad.shape
        x_win = window_partition(x_pad, self.window_size)
        cor_win = window_partition(cor_pad, self.window_size)

        nwB = x_win.shape[0]
        x_norm = self.norm1(x_win)

        x_reverse = torch.cat([x_norm[nwB//2:], x_norm[:nwB//2]])
        x_appearence, x_motion = self.attn(x_norm, x_reverse, cor_win, H, W, shift_mask)
        x_norm = x_norm + self.drop_path(x_appearence)

        x_back = x_norm
        x_back_win = window_reverse(x_back, self.window_size, Hw, Ww)
        x_motion = window_reverse(x_motion, self.window_size, Hw, Ww)
        
        if self.shift_size[0] or self.shift_size[1]:
            x_back_win = torch.roll(x_back_win, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x_motion = torch.roll(x_motion, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        x = depad_if_needed(x_back_win, x.size(), self.window_size).view(2*B, H * W, -1)
        x_motion = depad_if_needed(x_motion, cor.size(), self.window_size).view(2*B, H * W, -1)
            
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x, x_motion


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depths=2,act_layer=nn.PReLU):
        super().__init__()
        layers = []
        for i in range(depths):
            if i == 0:
                layers.append(nn.Conv2d(in_dim, out_dim, 3,1,1))
            else:
                layers.append(nn.Conv2d(out_dim, out_dim, 3,1,1))
            layers.extend([
                act_layer(out_dim),
            ])
        self.conv = nn.Sequential(*layers)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class CrossScalePatchEmbed(nn.Module):
    def __init__(self, in_dims=[16,32,64], embed_dim=768):
        super().__init__()
        base_dim = in_dims[0]

        layers = []
        for i in range(len(in_dims)):
            for j in range(2 ** i):
                layers.append(nn.Conv2d(in_dims[-1-i], base_dim, 3, 2**(i+1), 1+j, 1+j))
        self.layers = nn.ModuleList(layers)
        self.proj = nn.Conv2d(base_dim * len(layers), embed_dim, 1, 1)
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, xs):
        ys = []
        k = 0
        for i in range(len(xs)):
            for _ in range(2 ** i):
                ys.append(self.layers[k](xs[-1-i]))
                k += 1
        x = self.proj(torch.cat(ys,1))
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class EventFusion(nn.Module):
    """Standalone event fusion module."""

    def __init__(self, event_channels: Optional[int] = None, embed_dims: Optional[list] = None):
        super().__init__()
        self.event_channels = event_channels
        self.embed_dims = embed_dims or []
        self.linear_x = nn.ModuleDict()
        self.linear_cor: Optional[nn.Linear] = None

        if self.event_channels is not None:
            for dim in self.embed_dims:
                self.linear_x[str(dim)] = nn.Linear(self.event_channels, dim)
            self.linear_cor = nn.Linear(self.event_channels, 2)

    def _build_if_needed(self, event_channels: int, embed_dim: int, device: torch.device) -> None:
        key = str(embed_dim)
        if key not in self.linear_x:
            self.linear_x[key] = nn.Linear(event_channels, embed_dim).to(device)
        if self.linear_cor is None:
            self.linear_cor = nn.Linear(event_channels, 2).to(device)

    def flatten_event(self, event_feat: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """[B, Ce, H, W] -> [B, HW, Ce]."""
        if event_feat.shape[-2:] != (H, W):
            event_feat = F.interpolate(event_feat, size=(H, W), mode="bilinear", align_corners=False)
        return event_feat.flatten(2).transpose(1, 2).contiguous()

    def duplicate_for_2B(self, event_flat: torch.Tensor) -> torch.Tensor:
        """[B, HW, Ce] -> [2B, HW, Ce]."""
        # Event features are duplicated for bidirectional flow (frame1->frame2 and frame2->frame1).
        return torch.cat([event_flat, event_flat], dim=0)

    def forward(
        self,
        x: torch.Tensor,
        cor: torch.Tensor,
        event_feat: torch.Tensor,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            x: [2B, HW, C]
            cor: [2B, HW, 2]
            event_feat: [B, Ce, H, W]
        Outputs:
            fused_x: [2B, HW, C]
            fused_cor: [2B, HW, 2]
        """
        if x.shape[0] != 2 * event_feat.shape[0]:
            raise ValueError(
                "EventFusion expects x batch=2B and event batch=B, got {} and {}.".format(
                    x.shape[0], event_feat.shape[0]
                )
            )
        event_flat = self.flatten_event(event_feat, H, W)                 # [B, HW, Ce]
        event_2b = self.duplicate_for_2B(event_flat)                      # [2B, HW, Ce]
        self._build_if_needed(event_2b.shape[-1], x.shape[-1], x.device)
        projected_x = self.linear_x[str(x.shape[-1])](event_2b)           # Ce -> C
        projected_cor = self.linear_cor(event_2b)                         # Ce -> 2
        # Event injection into appearance (x)
        fused_x = x + projected_x
        # Event injection into motion prior (cor)
        fused_cor = cor + projected_cor
        return fused_x, fused_cor


class MotionFormer(nn.Module):
    def __init__(self, in_chans=3, embed_dims=[32, 64, 128, 256, 512], motion_dims=64, num_heads=[8, 16], 
                 mlp_ratios=[4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 2, 2, 6, 2], window_sizes=[11, 11], event_channels: Optional[int] = None, **kwarg):
        super().__init__()
        self.depths = depths
        self.num_stages = len(embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        self.conv_stages = self.num_stages - len(num_heads)

        for i in range(self.num_stages):
            if i == 0:
                block = ConvBlock(in_chans,embed_dims[i],depths[i])
            else:
                if i < self.conv_stages:
                    patch_embed = nn.Sequential(
                        nn.Conv2d(embed_dims[i-1], embed_dims[i], 3,2,1),
                        nn.PReLU(embed_dims[i])
                    )
                    block = ConvBlock(embed_dims[i],embed_dims[i],depths[i])
                else:
                    if i == self.conv_stages:
                        patch_embed = CrossScalePatchEmbed(embed_dims[:i],
                                                        embed_dim=embed_dims[i])
                    else:
                        patch_embed = OverlapPatchEmbed(patch_size=3,
                                                        stride=2,
                                                        in_chans=embed_dims[i - 1],
                                                        embed_dim=embed_dims[i])

                    block = nn.ModuleList([MotionFormerBlock(
                        dim=embed_dims[i], motion_dim=motion_dims[i], num_heads=num_heads[i-self.conv_stages], window_size=window_sizes[i-self.conv_stages], 
                        shift_size= 0 if (j % 2) == 0 else window_sizes[i-self.conv_stages] // 2,
                        mlp_ratio=mlp_ratios[i-self.conv_stages], qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer)
                        for j in range(depths[i])])

                    norm = norm_layer(embed_dims[i])
                    setattr(self, f"norm{i + 1}", norm)
                setattr(self, f"patch_embed{i + 1}", patch_embed)
            cur += depths[i]

            setattr(self, f"block{i + 1}", block)

        self.cor = {}
        # NEW: event fusion module (kept optional in forward path)
        self.event_fusion = EventFusion(
            event_channels=event_channels,
            embed_dims=embed_dims[self.conv_stages:],
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def get_cor(self, shape, device):
        k = (str(shape), str(device))
        if k not in self.cor:
            tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
                1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
            tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
                1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
            self.cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
        return self.cor[k]

    def freeze_backbone(self) -> None:
        """Strategy B: freeze conv backbone and keep last transformer stages + event_fusion trainable."""
        for p in self.parameters():
            p.requires_grad = False

        # Freeze: block1-3 and patch_embed1-3
        for idx in range(1, min(3, self.num_stages) + 1):
            block = getattr(self, f"block{idx}", None)
            patch_embed = getattr(self, f"patch_embed{idx}", None)
            if block is not None:
                for p in block.parameters():
                    p.requires_grad = False
            if patch_embed is not None:
                for p in patch_embed.parameters():
                    p.requires_grad = False

        # Freeze: CrossScalePatchEmbed (first transformer stage patch embedding).
        cross_idx = self.conv_stages + 1
        cross_embed = getattr(self, f"patch_embed{cross_idx}", None)
        if cross_embed is not None:
            for p in cross_embed.parameters():
                p.requires_grad = False

        # Keep trainable: last 1-2 transformer stages + corresponding norm layers.
        transformer_start = self.conv_stages + 1
        transformer_end = self.num_stages
        last_start = max(transformer_start, transformer_end - 1)
        for idx in range(last_start, transformer_end + 1):
            block = getattr(self, f"block{idx}", None)
            norm = getattr(self, f"norm{idx}", None)
            if block is not None:
                for p in block.parameters():
                    p.requires_grad = True
            if norm is not None:
                for p in norm.parameters():
                    p.requires_grad = True

        # Keep trainable: event_fusion
        for p in self.event_fusion.parameters():
            p.requires_grad = True

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, event_feat: Optional[torch.Tensor] = None):
        B = x1.shape[0] 
        x = torch.cat([x1, x2], 0)
        motion_features = []
        appearence_features = []
        xs = []
        for i in range(self.num_stages):
            motion_features.append([])
            patch_embed = getattr(self, f"patch_embed{i + 1}",None)
            block = getattr(self, f"block{i + 1}",None)
            norm = getattr(self, f"norm{i + 1}",None)
            if i < self.conv_stages:
                if i > 0:
                    x = patch_embed(x)
                x = block(x)
                xs.append(x)
            else:
                if i == self.conv_stages:
                    x, H, W = patch_embed(xs)
                else:
                    x, H, W = patch_embed(x)
                # EXISTING
                cor = self.get_cor((x.shape[0], H, W), x.device)
                # NEW: Event fusion point
                if event_feat is not None:
                    cor_tokens = cor.view(2 * B, H * W, 2)
                    x, cor_tokens = self.event_fusion(x, cor_tokens, event_feat, H, W)
                    cor = cor_tokens.view(2 * B, H, W, 2)
                for blk in block:
                    x, x_motion = blk(x, cor, H, W, B)
                    motion_features[i].append(x_motion.reshape(2*B, H, W, -1).permute(0, 3, 1, 2).contiguous())
                x = norm(x)
                x = x.reshape(2*B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                motion_features[i] = torch.cat(motion_features[i], 1)
            appearence_features.append(x)
        return appearence_features, motion_features


class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.reshape(B, C, -1).transpose(1, 2)

        return x


def feature_extractor(**kargs):
    model = MotionFormer(**kargs)
    return model
