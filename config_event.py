from functools import partial
import torch.nn as nn

from model.feature_extractor_events import feature_extractor as feature_extractor_events
from model import flow_estimation


def init_model_config_event(F=32, W=7, depth=[2, 2, 2, 4, 4], event_channels=5):
    return {
        'embed_dims':[F, 2*F, 4*F, 8*F, 16*F],
        'motion_dims':[0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'num_heads':[8*F//32, 16*F//32],
        'mlp_ratios':[4, 4],
        'qkv_bias':True,
        'norm_layer':partial(nn.LayerNorm, eps=1e-6),
        'depths':depth,
        'window_sizes':[W, W],
        'event_channels': event_channels,
    }, {
        'embed_dims':[F, 2*F, 4*F, 8*F, 16*F],
        'motion_dims':[0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'depths':depth,
        'num_heads':[8*F//32, 16*F//32],
        'window_sizes':[W, W],
        'scales':[4, 8, 16],
        'hidden_dims':[4*F, 4*F],
        'c':F,
    }


MODEL_CONFIG = {
    'LOGNAME': 'ours_event',
    'MODEL_TYPE': (feature_extractor_events, flow_estimation),
    'MODEL_ARCH': init_model_config_event(
        F=32,
        W=7,
        depth=[2, 2, 2, 4, 4],
        event_channels=5,
    ),
}
