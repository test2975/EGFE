import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.functional import Tensor
from torch.nn.modules.loss import _Loss as Loss
import torch.nn.functional as F

from sketch_model.configs import SketchModelConfig
from sketch_model.layers import LayerStructureEmbedding, build_transformer, SketchTransformer
from sketch_model.utils import NestedTensor


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SketchLayerClassifierModel(nn.Module):
    def __init__(
            self,
            config: SketchModelConfig,
            transformer: SketchTransformer,
    ):
        super().__init__()
        self.config = config
        self.hidden_dim = transformer.d_model
        self.structure_embed = LayerStructureEmbedding(
            config,
        )
        self.transformer: SketchTransformer = transformer
        self.cross_attn = nn.MultiheadAttention(
            self.hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.mlp = MLP(self.hidden_dim, self.hidden_dim, config.num_classes, 1)

    def with_pos_embeds(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
            self,
            x: torch.Tensor
    ):
        """
        Args:
            x (_type_): batch_img,  3 * 64 * 64
            batch_name, 32
            batch_bbox, 4
            batch_color, 4
            batch_class, 1
            mask, 1
        """
        # NOTE This is a patch for DateParall, so it's not elegant at all!
        # Ignore it please.
        mask = x[:, :, -1:].squeeze(dim=2).bool()
        batch_img = x[:, :, :-42].float()
        batch_img = batch_img.view(
            *batch_img.shape[:2], 3, int(math.sqrt(batch_img.shape[-1] // 3)),
            int(math.sqrt(batch_img.shape[-1] // 3)))
        batch_class = NestedTensor(x[:, :, -2:-1].squeeze(dim=2).int(), mask)
        batch_color = NestedTensor(x[:, :, -6:-2].float(), mask)
        batch_bbox = NestedTensor(x[:, :, -10:-6].float(), mask)
        batch_name = NestedTensor(x[:, :, -42:-10].int(), mask)
        batch_img = NestedTensor(batch_img, mask)

        # like DETR the x represents the features from backbone, shape is (b, c(256), len_seq)
        # like DETR pos represnts the position encodings from backbone, shape is (b,c,len_seq)
        # torch.Size([8, 176, 256]) torch.Size([8, 176, 256])
        # Embed
        if self.config.use_image:
            x, pos_embed, image_embeds = self.structure_embed(
                batch_img, batch_name, batch_bbox, batch_color, batch_class)
        else:
            x, pos_embed = self.structure_embed(
                batch_img, batch_name, batch_bbox, batch_color, batch_class)
        # Mask
        if self.config.use_mask:
            mask = batch_class.mask
        else:
            mask = None
        # Transformer
        x = self.transformer(x, mask, pos_embed)
        # Classifier
        class_embed = self.mlp(x)
        # class_embed = self.class_embed(x)  # TODO 验证用MLP
        return class_embed.softmax(dim=-1)


def build(config: SketchModelConfig) -> Tuple[SketchLayerClassifierModel, Loss]:
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu")
    transformer = build_transformer(config)
    model = SketchLayerClassifierModel(
        config,
        transformer,
    )
    class_weight = torch.FloatTensor(eval(config.class_weight))
    print("class_weight", class_weight)
    criterion = nn.CrossEntropyLoss(
        weight=class_weight, reduction='mean')
    criterion.to(device)
    return model, criterion
