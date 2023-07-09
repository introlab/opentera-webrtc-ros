from collections import OrderedDict

import torch


def load_checkpoint(model, file, strict=True, keys_to_remove=None):
    if keys_to_remove is None:
        keys_to_remove = []

    state_dict = torch.load(file, map_location=torch.device('cpu'))
    state_dict = OrderedDict([(k[7:], v) if k.startswith('module.') else (k, v) for k, v in state_dict.items()])

    for k in keys_to_remove:
        state_dict.pop(k, None)

    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        if not strict:
            _remove_size_mismatch_parameters(state_dict, e.args[0].splitlines()[1:])
            model.load_state_dict(state_dict, strict=strict)
        else:
            raise e


def _remove_size_mismatch_parameters(state_dict, size_mismatch_lines):
    for size_mismatch_line in size_mismatch_lines:
        parameter_end_index = size_mismatch_line.find(':')
        del state_dict[size_mismatch_line[19:parameter_end_index]]
