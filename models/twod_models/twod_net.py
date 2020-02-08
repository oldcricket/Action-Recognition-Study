import torch.nn as nn
import torch

# from .TRNmodule import return_TRN

from functools import partial
from .temporal_modeling import temporal_modeling_module, ConsensusModule

from .resnet import resnet

BACKBONE_MODEL_TABLE = {
    'resnet': resnet
}


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


#TODO: support TRN?

class LearnableTSMNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        args = DotDict(kwargs)
        self.args = args
        # set up temporal model
        temporal_module = partial(temporal_modeling_module, name=args.tsm_module,
                                  dw_conv=args.dw_conv,
                                  blending_frames=args.blending_frames,
                                  blending_method=args.blending_method)

        kwargs['temporal_module'] = temporal_module
        self.baseline_model = BACKBONE_MODEL_TABLE[args.backbone_net](**kwargs)
        self.num_frames = args.groups  # need to equal tsm_duration
        self.dropout = args.dropout
        self.modality = args.modality

        # TODO ??
        self.TRN_feature_dim = 256

        # new category
        # if args.consensus == 'avg':
        #     self.consensus = ConsensusModule(consensus_type='avg', dim=1)
        # else:
        #     # only do multiscale
        #     self.consensus = return_TRN('TRNmultiscale', self.TRN_feature_dim, self.tsm_duration, num_classes)

        # get the dim of feature vec
        feature_dim = getattr(self.baseline_model, 'fc').in_features
        # update the fc layer and initialize it
        self.prepare_baseline(args.consensus, feature_dim, args.num_classes)

    def prepare_baseline(self, consensus_type, feature_dim, num_classes):
        if consensus_type == 'avg':
            if self.dropout > 0.0:
                # replace the original fc layer as dropout layer
                setattr(self.baseline_model, 'fc', nn.Dropout(p=self.dropout))
                self.new_fc = nn.Linear(feature_dim, num_classes)
                nn.init.normal_(self.new_fc.weight, 0, 0.001)
                nn.init.constant_(self.new_fc.bias, 0)
            else:
                setattr(self.baseline_model, 'fc', nn.Linear(feature_dim, num_classes))
                nn.init.normal_(getattr(self.baseline_model, 'fc').weight, 0, 0.001)
                nn.init.constant_(getattr(self.baseline_model, 'fc').bias, 0)
        elif consensus_type == 'trn':
            assert self.dropout > 0.0
            # replace the original fc layer as dropout layer
            setattr(self.baseline_model, 'fc', nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, self.TRN_feature_dim)
            nn.init.normal_(self.new_fc.weight, 0, 0.001)
            nn.init.constant_(self.new_fc.bias, 0)
        else:
            raise ValueError('consensus type %s is not supported' % consensus_type)

    def forward(self, x):
        n, c_t, h, w = x.shape
        batched_input = x.view(n * self.num_frames, c_t // self.num_frames, h, w)
        base_out = self.baseline_model(batched_input)
        if self.dropout > 0.0:
            base_out = self.new_fc(base_out)
        n_t, c = base_out.shape
        base_out = base_out.view(n, -1, c)
        # average all frames
        # out = self.consensus(base_out)
        # out = out.squeeze(1)
        out = torch.mean(base_out, dim=1)
        # dim of out: [N, num_classes, 1, 1]
        return out

    def mean(self, modality='rgb'):
        return self.baseline_model.mean(modality)

    def std(self, modality='rgb'):
        return self.baseline_model.std(modality)

    @property
    def network_name(self):
        if self.args.tsm_module is None:
            name = "{}".format(self.baseline_model.network_name)
        else:
            name = "{}-b{}-{}{}-{}".format(self.args.tsm_module, self.args.blending_frames,
                                           self.args.blending_method,
                                           "" if self.args.dw_conv else "-allc",
                                           self.baseline_model.network_name)
        return name
