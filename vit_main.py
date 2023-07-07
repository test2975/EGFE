from sketch_model.model import build, SketchLayerClassifierModel
import datetime
import json
import math
import os
import random
import sys
import time
import urllib.parse
from dataclasses import fields
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy
import numpy as np
import requests
import torch
from torch import nn
from torch.nn.modules.loss import _Loss as Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, PreTrainedTokenizer
from vision_transformer import vit, vit2
from sketch_model.configs import SketchModelConfig, config_with_arg, ModelConfig
from sketch_model.datasets import build_dataset
from sketch_model.utils import misc as utils, f1score, r2score, accuracy_simple, creat_confusion_matrix, precision, recall, confusion_matrix
from sketch_model.utils import NestedTensor
from sketch_model.utils.utils import flatten_input_for_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def init_device(config: SketchModelConfig) -> torch.device:
    if config.device == 'cuda' and not torch.cuda.is_available():
        config.device = 'cpu'
    device = torch.device(config.device)
    return device


def init_config(
        config: SketchModelConfig) -> Tuple[SketchModelConfig, Dict[str, Any]]:
    '''
    fix the seed for reproducibility
    if resume, loading config checkpoint
    '''

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    checkpoint = None
    if config.resume:
        checkpoint = torch.load(config.resume, map_location='cpu')
        saved_config: SketchModelConfig = checkpoint['config']
        config.start_epoch = checkpoint['epoch'] + 1
        for field in fields(ModelConfig):
            # override the current config by using saved config
            config.__setattr__(field.name,
                               saved_config.__getattribute__(field.name))
    return config, checkpoint


def init_model(
    config: SketchModelConfig, checkpoint: Dict[str, Any], device: torch.device
) -> Tuple[PreTrainedTokenizer, SketchLayerClassifierModel, Loss, Optimizer,
           _LRScheduler]:
    model = vit2.ViT(image_size=256,
                    patch_size=32,
                    num_classes=3,
                    dim=1024,
                    depth=6,
                    heads=16,
                    mlp_dim=2048,
                    dropout=0.1,
                    emb_dropout=0.1)
    class_weight = torch.FloatTensor(eval(config.class_weight))
    criterion = nn.CrossEntropyLoss(
        weight=class_weight, reduction='mean')
    criterion.to(device)
    model.to(device)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name)
    tokenizer.model_max_length = config.max_name_length
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id

    model_without_ddp = model
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters(
        )]}
    ]
    optimizer = torch.optim.AdamW(param_dicts,
                                  lr=config.lr,
                                  weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)

    if checkpoint is not None:
        print("Loading Checkpoint...")
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    return (tokenizer, model, model_without_ddp, criterion, optimizer, lr_scheduler)


def main(config: SketchModelConfig):
    utils.init_distributed_mode(config)

    print(f"Batch Size: {config.batch_size * config.world_size}")
    print("Loading Config...")
    device = init_device(config)  # cuda if gpu available else cpu
    config, checkpoint = init_config(
        config
    )  # loading saved config and checkpoint if resume else CK is None

    print("Loading Model...")
    tokenizer, model, model_without_ddp, criterion, optimizer, lr_scheduler = init_model(
        config, checkpoint, device)

    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print('number of params:', n_parameters)

    print("Loading Test Dataset...")
    dataset_val = build_dataset(config.test_index_json,
                                Path(config.test_index_json).parent.__str__(),
                                tokenizer,
                                cache_dir=config.cache_dir,
                                use_cache=config.use_cache,
                                remove_text=config.remove_text)
    if config.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val,
                                 config.batch_size,
                                 sampler=sampler_val,
                                 drop_last=False,
                                 collate_fn=utils.collate_fn,
                                 num_workers=config.num_workers)
    if config.evaluate:
        test_stats = evaluate(config, model, criterion, data_loader_val,
                              device)

        if utils.is_main_process():
            print("Evaluating macro")
            print(test_stats[1])
            with open('./log.txt', 'a') as f:
                f.write(json.dumps(test_stats[1]) + "\n")
        return

    print("Loading Train Dataset...")
    dataset_train = build_dataset(config.train_index_json,
                                  Path(
                                      config.test_index_json).parent.__str__(),
                                  tokenizer,
                                  cache_dir=config.cache_dir,
                                  use_cache=config.use_cache,
                                  remove_text=config.remove_text)
    if config.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = RandomSampler(dataset_train)

    batch_sampler_train = BatchSampler(sampler_train,
                                       config.batch_size,
                                       drop_last=True)
    data_loader_train = DataLoader(dataset_train,
                                   batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn,
                                   num_workers=config.num_workers)

    output_dir = Path(
        config.output_dir
    ) / f'{config.task_name}-{time.strftime("%d-%H%M", time.localtime())}'
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    tensorboard_dir = output_dir / 'tensorboard'
    os.makedirs(tensorboard_dir, exist_ok=True)
    if utils.is_main_process:
        config.save(output_dir / 'config.json')

    print("Start training")
    start_time = time.time()
    writer = SummaryWriter(f'{str(tensorboard_dir)}')
    best_train_acc, best_train_f1, best_test_acc, best_test_f1, best_test_precision, best_test_recall = [
        0
    ] * 6
    for epoch in range(config.start_epoch, config.epochs):
        if config.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(config, model, criterion,
                                      data_loader_train, optimizer, device,
                                      epoch, config.clip_max_norm)
        lr_scheduler.step()
        y_pred, y_true = train_stats[0]
        train_stats = train_stats[1]
        # save the checkpoint
        if config.output_dir:
            checkpoint_paths = [checkpoint_dir / 'latest.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % config.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(checkpoint_dir /
                                        f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'config': config,
                    }, checkpoint_path)

        test_stats = evaluate(config, model, criterion, data_loader_val,
                              device)
        y_pred, y_true = test_stats[0]
        test_stats = test_stats[1]
        log_stats = {
            **{f'train_{k}': v
               for k, v in train_stats.items()},
            **{f'test_{k}': v
               for k, v in test_stats.items()}, 'epoch': epoch,
            'n_parameters': n_parameters
        }

        best_train_acc, best_train_f1, best_test_acc, best_test_f1,\
            best_test_precision, best_test_recall = np.maximum(
                (best_train_acc, best_train_f1, best_test_acc,
                 best_test_f1, best_test_precision, best_test_recall),
                (train_stats['acc'], train_stats['f1'], test_stats['acc'],
                 test_stats['f1'], test_stats['precision'], test_stats['recall']))
        if utils.is_main_process():
            writer.add_figure("test/Confusion matrix",
                              creat_confusion_matrix(y_true, y_pred), epoch)
            writer.add_scalar('train/loss', train_stats['loss'], epoch)
            writer.add_scalar('train/acc', train_stats['acc'], epoch)
            writer.add_scalar('train/f1', train_stats['f1'], epoch)
            writer.add_scalar('train/r2', train_stats['r2'], epoch)
            writer.add_scalar('test/loss', test_stats['loss'], epoch)
            writer.add_scalar('test/acc', test_stats['acc'], epoch)
            writer.add_scalar('test/f1', test_stats['f1'], epoch)
            writer.add_scalar('test/r2', test_stats['r2'], epoch)
            writer.add_scalar('test/precision', test_stats['precision'], epoch)
            writer.add_scalar('test/recall', test_stats['recall'], epoch)
            writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'],
                              epoch)

        if config.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    writer.flush()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print('Best test acc: {}'.format(best_test_acc))
    print('Best test f1: {}'.format(best_test_f1))
    print('Best test precision: {}'.format(best_test_precision))
    print('Best test recall: {}'.format(best_test_recall))
    print('Best train acc: {}'.format(best_train_acc))
    print('Best train f1: {}'.format(best_train_f1))
    # for get the train message in my mobile device
    sct_key = os.environ.get("SCT_KEY")
    if sct_key and utils.is_main_process():
        title = f'{config.task_name} finished'
        content = f'Training {config.task_name} finished, total time:{total_time_str}, best test acc:{best_test_acc}, best test f1:{best_test_f1}, best train acc:{best_train_acc}, best train f1:{best_train_f1}'
        res = requests.get(
            f"https://sctapi.ftqq.com/{sct_key}.send?title={urllib.parse.quote_plus(title)}&desp={urllib.parse.quote_plus(content)}"
        )
        print('response:\n', res.text)


def train_one_epoch(
    config: SketchModelConfig,
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader[str],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'acc', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'f1', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'precision', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'recall', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'r2', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Task:{config.task_name} Epoch: [{epoch}]'
    print_freq = 10
    y_pred, y_true = [], []
    for (batch_img, _, _, _, _), targets in metric_logger.log_every(
            dataloader, print_freq, header):

        batch_img: NestedTensor = batch_img.to(device)  # [b, seqlen, c, w, h]
        outputs = model(batch_img)

        targets = [t.to(device) for t in targets]  # [[120] ... []]

        batch_ce_loss = torch.tensor(0.0, device=device)
        acc, f1, r2, precision_score, recall_score = 0, 0, 0, 0, 0
        for i in range(len(targets)):
            packed = outputs[i][:len(targets[i])]  # [120, 4]
            ce_loss = criterion(packed, targets[i])
            batch_ce_loss += ce_loss
            pred = packed.max(-1)[1]
            # for confusion matrix
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(targets[i].cpu().numpy())
            # for logger
            acc += accuracy_simple(pred, targets[i])
            f1 += f1score(pred, targets[i])
            r2 += r2score(pred, targets[i])
            precision_score += precision(pred, targets[i])
            recall_score += recall(pred, targets[i])
        acc, f1, r2, precision_score, recall_score = numpy.array(
            [acc, f1, r2, precision_score, recall_score]) / len(targets)

        if not math.isfinite(batch_ce_loss):
            print("Loss is {}, stopping training".format(batch_ce_loss))
            sys.exit(1)

        optimizer.zero_grad()
        batch_ce_loss.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        metric_logger.update(acc=acc,
                             f1=f1,
                             r2=r2,
                             precision=precision_score,
                             recall=recall_score)
        metric_logger.update(loss=batch_ce_loss.detach())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg
             for k, meter in metric_logger.meters.items()}
    print("[Train] Averaged stats:", stats)
    return [(y_pred, y_true),
            stats]


@torch.no_grad()
def evaluate(config: SketchModelConfig,
             model: nn.Module,
             criterion: nn.Module,
             dataloader: DataLoader,
             device: torch.device,
             eval_model: str = "macro"):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'acc', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'f1', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'r2', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'precision', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'recall', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Task:{config.task_name} Test:'
    print_freq = 10
    y_pred, y_true = [], []
    for (batch_img, _, _, _, _), targets in metric_logger.log_every(
            dataloader, print_freq, header):
        batch_img = batch_img.to(device)
        targets = [t.to(device) for t in targets]

        outputs = model(batch_img)
        batch_ce_loss = torch.tensor(0.0, device=device)
        acc, f1, r2, precision_score, recall_score = 0, 0, 0, 0, 0
        for i in range(len(targets)):
            packed = outputs[i][:len(targets[i])]
            ce_loss = criterion(packed, targets[i])
            batch_ce_loss += ce_loss
            pred = packed.max(-1)[1]
            y_pred.extend(pred.cpu().numpy())
            y_true.extend(targets[i].cpu().numpy())
            acc += accuracy_simple(pred, targets[i])
            r2 += r2score(pred, targets[i])
            precision_score += precision(pred, targets[i], average=eval_model)
            recall_score += recall(pred, targets[i], average=eval_model)
        acc, r2, precision_score, recall_score = numpy.array(
            [acc, r2, precision_score, recall_score]) / len(targets)
        f1 = 2 * precision_score * recall_score / (precision_score +
                                                   recall_score)
        metric_logger.update(acc=acc,
                             f1=f1,
                             r2=r2,
                             precision=precision_score,
                             recall=recall_score)
        metric_logger.update(loss=batch_ce_loss.detach())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg
             for k, meter in metric_logger.meters.items()}
    print("[TEST] Averaged stats:", stats)
    return [(y_pred, y_true),
            stats]


if __name__ == '__main__':
    main(config_with_arg())
