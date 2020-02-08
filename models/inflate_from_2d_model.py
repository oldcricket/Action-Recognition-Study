import torch
from collections import OrderedDict


def inflate_from_2d_model(state_dict_2d, state_dict_3d, skipped_keys=None, inflated_dim=2):

    if skipped_keys is None:
        skipped_keys = []

    missed_keys = []
    new_keys = []
    for old_key in state_dict_2d.keys():
        if old_key not in state_dict_3d.keys():
            missed_keys.append(old_key)
    for new_key in state_dict_3d.keys():
        if new_key not in state_dict_2d.keys():
            new_keys.append(new_key)
    print("Missed tensors: {}".format(missed_keys))
    print("New tensors: {}".format(new_keys))
    print("Following layers will be skipped: {}".format(skipped_keys))

    state_d = OrderedDict()
    unused_layers = [k for k in state_dict_2d.keys()]
    uninitialized_layers = [k for k in state_dict_3d.keys()]
    initialized_layers = []
    for key, value in state_dict_2d.items():
        skipped = False
        for skipped_key in skipped_keys:
            if skipped_key in key:
                skipped = True
                break
        if skipped:
            continue
        new_value = value
        # only inflated conv's weights
        if key in state_dict_3d:
            # TODO: a better way to identify conv layer?
            # if 'conv.weight' in key or \
            #         'conv1.weight' in key or 'conv2.weight' in key or 'conv3.weight' in key or \
            #         'downsample.0.weight' in key:
            if value.ndimension() == 4 and 'weight' in key:
                value = torch.unsqueeze(value, inflated_dim)
                # value.unsqueeze_(inflated_dim)
                repeated_dim = torch.ones(state_dict_3d[key].ndimension(), dtype=torch.int)
                repeated_dim[inflated_dim] = state_dict_3d[key].size(inflated_dim)
                new_value = value.repeat(repeated_dim.tolist())
            state_d[key] = new_value
            initialized_layers.append(key)
            uninitialized_layers.remove(key)
            unused_layers.remove(key)

    print("Initialized layers: {}".format(initialized_layers))
    print("Uninitialized layers: {}".format(uninitialized_layers))
    print("Unused layers: {}".format(unused_layers))

    return state_d


def convert_rgb_model_to_others(state_dict, input_channels, ks=7):
    new_state_dict = {}
    for key, value in state_dict.items():
        if "conv1.weight" in key:
            o_c, in_c, k_h, k_w = value.shape
        else:
            o_c, in_c, k_h, k_w = 0, 0, 0, 0
        if in_c == 3 and k_h == ks and k_w == ks:
            # average the weights and expand to all channels
            new_shape = (o_c, input_channels, k_h, k_w)
            new_value = value.mean(dim=1, keepdim=True).expand(new_shape).contiguous()
        else:
            new_value = value
        new_state_dict[key] = new_value
    return new_state_dict