#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
import os.path as osp
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def inference_on_dataloader(model, data_loader, dataset_name, tmpdir):
    from detectron2.utils.comm import get_world_size
    import time
    import datetime
    from detectron2.utils.logger import log_every_n_seconds
    from detectron2.structures import Instances
    import pickle
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))
    total = len(data_loader)  # inference data loader must have a fixed length

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    results = []
    with torch.no_grad():
        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            # forward progress
            ##########
            # hack to use submodule in ddp : model.module
            images = model.preprocess_image(inputs)
            feats = model.backbone(images.tensor)
            proposals, _ = model.proposal_generator(images, feats, None)
            for p, ori_input, img_size in zip(
                proposals, inputs, images.image_sizes
            ):
                def _post_process(p, output_height, output_width):
                    # rescale proposals and append objectness_logits to each proposals
                    # output : list [N, K, 5]
                    if isinstance(output_height, torch.Tensor):
                        # Converts integer tensors to float temporaries to ensure true
                        # division is performed when computing scale_x and scale_y.
                        output_width_tmp = output_width.float()
                        output_height_tmp = output_height.float()
                        new_size = torch.stack([output_height, output_width])
                    else:
                        new_size = (output_height, output_width)
                        output_width_tmp = output_width
                        output_height_tmp = output_height

                    scale_x, scale_y = (
                        output_width_tmp / p.image_size[1],
                        output_height_tmp / p.image_size[0],
                    )

                    results = Instances(new_size, **p.get_fields())
                    output_boxes = results.proposal_boxes
                    output_boxes.scale(scale_x, scale_y)
                    output_boxes.clip(results.image_size)
                    results = results[output_boxes.nonempty()]
                    ret = torch.cat([results.proposal_boxes.tensor, 
                                    results.objectness_logits[:, None]], dim=1)
                    return ret.cpu().numpy()
                height = ori_input.get("height", img_size[0])
                width = ori_input.get("width", img_size[1])
                ret = _post_process(p, height, width)
                results.append({"img_id":ori_input.get("image_id") , "proposals": ret})      
            ##########
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # log each 1000 iter
            if idx % 2000 == 0:
                tmppath = '/data/tmp'
                mkdir_or_exist(tmppath)
                outpath = osp.join(tmppath, "{}_rpn_proposals_train{}.pkl".format(dataset_name, idx))
                with open(outpath, 'wb') as ff:
                    pickle.dump(results, ff, protocol=2)
                    logger.info('save proposals on {} in {}' .format(dataset_name, outpath))  
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s / iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s / iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s / iter. "
                        f"Total: {total_seconds_per_iter:.4f} s / iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()
    
    
    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    # gather results
    if num_devices == 1:
        return results
    else: 
        gather_time_tik = time.perf_counter()
        results = collect_results_cpu(results, total, tmpdir)  
        gather_time_tok = time.perf_counter()
        logger.info("Totol gather time: {}".format(
            str(datetime.timedelta(seconds=int(gather_time_tok-gather_time_tik))))) 
        return results

def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size



def collect_results_cpu(result_part, size, tmpdir=None):
    import shutil
    import pickle
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        tmpdir = '/data/detpro/tmp'
    mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    part_file_path = osp.join(tmpdir, f'part_{rank}.pkl')
    with open(part_file_path, 'wb') as ff:
        pickle.dump(result_part, ff, protocol=2)
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file_path = osp.join(tmpdir, f'part_{i}.pkl')
            with open(part_file_path, 'rb') as ff:
                part_list.append(pickle.load(ff))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results



def do_forward(cfg, model, outdir='/data/detpro/proposals', tmpdir='/data/detpro/tmp'):
    import pickle
    model.eval()
    logger = logging.getLogger(__name__)
    # build dataset
    for dataset_name in cfg.DATASETS.TEST:
        logger.info('Inference on {}'.format(dataset_name))
        clip_batchsize = cfg.SOLVER.IMS_PER_BATCH
        data_loader = build_detection_test_loader(cfg, dataset_name, clip_batch_size = clip_batchsize)
        results = inference_on_dataloader(model, data_loader, dataset_name, tmpdir)
        mkdir_or_exist(outdir)
        outpath = osp.join(outdir, '{}_rpn_proposals_train.pkl'.format(dataset_name))
        with open(outpath, 'wb') as ff:
            pickle.dump(results, ff, protocol=2)
        logger.info('save all proposals on {} in {}' .format(dataset_name, outpath))    
    return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume)
    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
    
    do_forward(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
