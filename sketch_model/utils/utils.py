# from sketch_model.utils.misc import NestedTensor
import torch
from .misc import NestedTensor


def flatten_input_for_model(batch_img: NestedTensor, batch_name: NestedTensor,
                            batch_bbox: NestedTensor,
                            batch_color: NestedTensor,
                            batch_class: NestedTensor):
    # NOTE This is a patch for DateParall, so it's not elegant at all!
    # Ignore it please.
    return torch.cat([
        batch_img.view(*batch_img.size()[:2], -1), batch_name.tensors,
        batch_bbox.tensors, batch_color.tensors,
        batch_class.unsqueeze(2),
        batch_class.mask.unsqueeze(2)
    ],
                     dim=2)
