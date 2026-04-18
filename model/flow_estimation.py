import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .warplayer import warp
from .refine import *

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class Head(nn.Module):
    def __init__(self, in_planes, scale, c, in_else=17):
        super(Head, self).__init__()
        self.upsample = nn.Sequential(nn.PixelShuffle(2), nn.PixelShuffle(2))
        self.scale = scale
        self.conv = nn.Sequential(
                                  conv(in_planes*2 // (4*4) + in_else, c),
                                  conv(c, c),
                                  conv(c, 5),
                                  )  

    def forward(self, motion_feature, x, flow): # /16 /8 /4
        motion_feature = self.upsample(motion_feature) #/4 /2 /1
        if self.scale != 4:
            x = F.interpolate(x, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False)
        if flow != None:
            if self.scale != 4:
                flow = F.interpolate(flow, scale_factor = 4. / self.scale, mode="bilinear", align_corners=False) * 4. / self.scale
            x = torch.cat((x, flow), 1)
        x = self.conv(torch.cat([motion_feature, x], 1))
        if self.scale != 4:
            x = F.interpolate(x, scale_factor = self.scale // 4, mode="bilinear", align_corners=False)
            flow = x[:, :4] * (self.scale // 4)
        else:
            flow = x[:, :4]
        mask = x[:, 4:5]
        return flow, mask

    
class MultiScaleFlow(nn.Module):
    def __init__(self, backbone, **kargs):
        super(MultiScaleFlow, self).__init__()
        self.flow_num_stage = len(kargs['hidden_dims'])
        self.feature_bone = backbone
        self.block = nn.ModuleList([Head( kargs['motion_dims'][-1-i] * kargs['depths'][-1-i] + kargs['embed_dims'][-1-i], 
                            kargs['scales'][-1-i], 
                            kargs['hidden_dims'][-1-i],
                            6 if i==0 else 17) 
                            for i in range(self.flow_num_stage)])
        self.unet = Unet(kargs['c'] * 2)
        # Inference-only flow visualization (disabled by default).
        self._flow_vis_cfg = {
            "enabled": False,
            "save_path": None,
            "seq_name": None,
            "idx": None,
            "max_samples": 30,
            "max_batch_items": None,
        }

    def configure_flow_visualization(
        self,
        enabled=False,
        save_path=None,
        seq_name=None,
        idx=None,
        max_samples=30,
        max_batch_items=None,
    ):
        self._flow_vis_cfg.update(
            {
                "enabled": bool(enabled),
                "save_path": save_path,
                "seq_name": seq_name,
                "idx": idx,
                "max_samples": int(max_samples),
                "max_batch_items": max_batch_items,
            }
        )

    @torch.no_grad()
    def _save_flow_visualization_if_needed(self, flow):
        cfg = self._flow_vis_cfg
        if not cfg.get("enabled", False):
            return

        idx = cfg.get("idx", None)
        if idx is None or idx >= cfg.get("max_samples", 30):
            return

        save_path = cfg.get("save_path", None)
        seq_name = cfg.get("seq_name", None)
        if not save_path or not seq_name:
            return

        from torchvision.utils import flow_to_image, save_image

        flow_0 = flow[:, :2]
        flow_1 = flow[:, 2:4]
        flow_0_imgs = flow_to_image(flow_0).float() / 255.0
        flow_1_imgs = flow_to_image(flow_1).float() / 255.0

        flow_vis_dir = os.path.join(save_path, "flow visualization", seq_name)
        os.makedirs(flow_vis_dir, exist_ok=True)

        bsz = flow.size(0)
        max_batch_items = cfg.get("max_batch_items", None)
        if max_batch_items is not None:
            bsz = min(bsz, int(max_batch_items))

        for b in range(bsz):
            if b == 0:
                name_0 = "flow_{:04d}_I0_to_t.png".format(idx)
                name_1 = "flow_{:04d}_I1_to_t.png".format(idx)
            else:
                name_0 = "flow_{:04d}_b{:02d}_I0_to_t.png".format(idx, b)
                name_1 = "flow_{:04d}_b{:02d}_I1_to_t.png".format(idx, b)

            save_image(flow_0_imgs[b], os.path.join(flow_vis_dir, name_0))
            save_image(flow_1_imgs[b], os.path.join(flow_vis_dir, name_1))

    def warp_features(self, xs, flow):
        y0 = []
        y1 = []
        B = xs[0].size(0) // 2
        for x in xs:
            y0.append(warp(x[:B], flow[:, 0:2]))
            y1.append(warp(x[B:], flow[:, 2:4]))
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        return y0, y1

    def calculate_flow(self, imgs, timestep, af=None, mf=None, event_feat=None):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        B = img0.size(0)
        flow, mask = None, None
        # appearence_features & motion_features
        if (af is None) or (mf is None):
            af, mf = self.feature_bone(img0, img1, event_feat=event_feat)
        for i in range(self.flow_num_stage):
            t = torch.full(mf[-1-i][:B].shape, timestep, dtype=torch.float).cuda()
            if flow != None:
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                flow_, mask_ = self.block[i](
                    torch.cat([t*mf[-1-i][:B],(1-t)*mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1),
                    torch.cat((img0, img1, warped_img0, warped_img1, mask), 1),
                    flow
                    )
                flow = flow + flow_
                mask = mask + mask_
            else:
                flow, mask = self.block[i](
                    torch.cat([t*mf[-1-i][:B],(1-t)*mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1),
                    torch.cat((img0, img1), 1),
                    None
                    )

        return flow, mask

    def coraseWarp_and_Refine(self, imgs, af, flow, mask):
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        c0, c1 = self.warp_features(af, flow)
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        mask_ = torch.sigmoid(mask)
        merged = warped_img0 * mask_ + warped_img1 * (1 - mask_)
        pred = torch.clamp(merged + res, 0, 1)
        return pred


    # Actually consist of 'calculate_flow' and 'coraseWarp_and_Refine'
    def forward(self, x, timestep=0.5, event_feat=None):
        img0, img1 = x[:, :3], x[:, 3:6]
        B = x.size(0)
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        # appearence_features & motion_features
        af, mf = self.feature_bone(img0, img1, event_feat=event_feat)
        for i in range(self.flow_num_stage):
            t = torch.full(mf[-1-i][:B].shape, timestep, dtype=torch.float).cuda()
            if flow != None:
                flow_d, mask_d = self.block[i]( torch.cat([t*mf[-1-i][:B], (1-timestep)*mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1), 
                                                torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow)
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = self.block[i]( torch.cat([t*mf[-1-i][:B], (1-t)*mf[-1-i][B:],af[-1-i][:B],af[-1-i][B:]],1), 
                                            torch.cat((img0, img1), 1), None)
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append(warped_img0 * mask_list[i] + warped_img1 * (1 - mask_list[i]))
        
        c0, c1 = self.warp_features(af, flow)
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        pred = torch.clamp(merged[-1] + res, 0, 1)
        self._save_flow_visualization_if_needed(flow_list[-1])
        return flow_list, mask_list, merged, pred
