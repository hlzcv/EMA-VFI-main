import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam

from model.loss import *
import config_event as cfg


class EventModel:
    def __init__(self, local_rank):
        backbonetype, multiscaletype = cfg.MODEL_CONFIG['MODEL_TYPE']
        backbonecfg, multiscalecfg = cfg.MODEL_CONFIG['MODEL_ARCH']
        self.net = multiscaletype(backbonetype(**backbonecfg), **multiscalecfg)
        self.name = cfg.MODEL_CONFIG['LOGNAME']
        self.device()

        self._freeze_backbone_for_strategy_b()
        self.optimG = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=2e-4)
        self.lap = LapLoss()

        if local_rank != -1:
            self.net = DDP(self.net, device_ids=[local_rank], output_device=local_rank)

    def _get_feature_bone(self):
        if isinstance(self.net, DDP):
            return self.net.module.feature_bone
        return self.net.feature_bone

    def _freeze_backbone_for_strategy_b(self):
        feature_bone = self._get_feature_bone()
        if hasattr(feature_bone, 'freeze_backbone'):
            feature_bone.freeze_backbone()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def device(self):
        self.net.to(torch.device('cuda'))

    def load_model(self, name=None, rank=0):
        def convert(param):
            return {
                k.replace('module.', ''): v
                for k, v in param.items()
                if 'module.' in k and 'attn_mask' not in k and 'HW' not in k
            }

        if rank <= 0:
            if name is None:
                name = 'ours'  # load pretrained RGB checkpoint by default
            self.net.load_state_dict(convert(torch.load(f'ckpt/{name}.pkl')), strict=False)

        # Re-apply freeze and optimizer after loading weights.
        self._freeze_backbone_for_strategy_b()
        self.optimG = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=2e-4)

    def save_model(self, rank=0):
        if rank == 0:
            torch.save(self.net.state_dict(), f'ckpt/{self.name}.pkl')

    def update(self, imgs, gt, event_feat=None, learning_rate=0, training=True):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate

        if training:
            self.train()
            flow, mask, merged, pred = self.net(imgs, event_feat=event_feat)
            loss_l1 = (self.lap(pred, gt)).mean()
            for merge in merged:
                loss_l1 += (self.lap(merge, gt)).mean() * 0.5

            self.optimG.zero_grad()
            loss_l1.backward()
            self.optimG.step()
            return pred, loss_l1

        self.eval()
        with torch.no_grad():
            flow, mask, merged, pred = self.net(imgs, event_feat=event_feat)
            return pred, 0
