import enum
from dataclasses import dataclass
from os import PathLike
from typing import Optional

from fastclasses_json import dataclass_json, JSONMixin

from sketch_model.datasets.dataset import LAYER_CLASS_MAP


class Aggregation(enum.Enum):
    CONCAT = 0
    SUM = 1


class PosPattern(enum.Enum):
    """
    Enum class for the possible position patterns.
    """
    ONE = 0
    FOUR = 1
    TWO = 2


class SentenceMethod(enum.Enum):
    SUM = 0
    MAX = 1
    MEAN = 2


@dataclass
class TransformerConfig:
    # fixed the hyperparam
    enc_layers: int = 6
    dim_feedforward: int = 2048
    hidden_dim: int = 256
    dropout: float = 0.2
    nheads: int = 8
    pre_norm: bool = True
    use_mask: bool = True


@dataclass
class DatasetConfig:
    train_index_json: str = '/media/sda1/cyn-workspace/sketch_transformer_dataset/index_train.json'
    test_index_json: str = '/media/sda1/cyn-workspace/sketch_transformer_dataset/index_test.json'
    lazy_load: bool = False
    remove_text: bool = True
    use_fullimage: bool = False


@dataclass
class SaveConfig:
    task_name: str = 'sketch_transformer'
    output_dir: str = './work_dir'
    cache_dir: str = './cache_dir'
    use_cache: bool = False
    resume: Optional[str] = None


@dataclass
class DeviceConfig:
    device: str = 'cuda'
    num_workers: int = 2


@dataclass
class DistributedConfig:
    rank: int = 0
    world_size: int = 4
    dist_url: str = "env://"
    gpu: int = 0
    distributed: bool = True
    dist_backend: str = "nccl"


@dataclass
class InitConfig:
    # fixed the hyperparam
    seed: int = 42
    start_epoch: int = 0
    epochs: int = 300
    batch_size: int = 2
    evaluate: bool = False
    eval_model: str = "macro"


@dataclass
class LRConfig:
    # fixed the hyperparam
    lr: float = 1e-4
    lr_backbone: float = lr * 0.1
    lr_drop: int = 200


@dataclass
class ModelConfig(TransformerConfig, LRConfig):
    # fixed the hyperparam
    weight_decay: float = 1e-4
    clip_max_norm: float = 0.1

    tokenizer_name: str = 'bert-base-chinese'
    max_name_length: int = 32
    num_classes: int = 3
    max_seq_length: int = 200

    class_types: int = len(LAYER_CLASS_MAP)
    pos_freq: int = 64
    pos_pattern: PosPattern = PosPattern.ONE
    sentence_method: SentenceMethod = SentenceMethod.SUM
    aggregation: Aggregation = Aggregation.SUM

    use_image: bool = True
    use_name: bool = True
    use_color: bool = True
    use_class: bool = True
    use_position: bool = True
    position_embedding = 'v4'

    vocab_size: int = 21128
    pad_token_id: int = 0

    add_mlp: bool = False
    class_weight: str = '[1,45,20]'


@dataclass_json
@dataclass
class SketchModelConfig(JSONMixin, DatasetConfig, SaveConfig, DeviceConfig,
                        InitConfig, ModelConfig, DistributedConfig):

    def save(self, path: PathLike):
        open(path, 'w').write(self.to_json())
