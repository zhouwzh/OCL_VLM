from argparse import ArgumentParser
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch as pt
import tqdm

from object_centric_bench.datum import DataLoader
from object_centric_bench.learn import MetricWrap
from object_centric_bench.model import ModelWrap
from object_centric_bench.utils import Config, build_from_config
# import torch._dynamo
# torch._dynamo.config.capture_scalar_outputs = True
pt.use_deterministic_algorithms(False)

from torch.nn import DataParallel as DP

import torch.nn as nn

class DPInputAdaptor(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped # ModelWrap

    def forward(self, **batch):
        # batch: {'image': tensor, ...}
        # fn(**batch): fn(image=x,...)
        # ModelWrap.forward expect dict/Config as the only positional arg
        return self.wrapped(Config(batch))


def train_epoch(args,pack):
    # import pdb; pdb.set_trace()
    t0 = time.time()
    pack.model.train()
    [_.before_epoch(**pack) for _ in pack.callback_t]

    for batch in tqdm.tqdm(pack.dataset_t):
        if pack.step_count + 1 > pack.total_step:
            break
        pack.batch = {k: v.cuda() for k, v in batch.items()}

        [_.before_step(**pack) for _ in pack.callback_t]

        with pt.autocast("cuda", enabled=True):
            # import pdb; pdb.set_trace()
            if args.use_dp:
                pack.output = pack.model(**pack.batch)
            else:
                pack.output = pack.model(pack.batch)
            [_.after_forward(**pack) for _ in pack.callback_t]
            pack.loss = pack.loss_fn(**pack)
            if args.use_dp:
                pack.loss = {k: v.mean() for k,v in pack.loss.items()}
            # pack.loss["vq_loss"] = pack.output.vq_loss
            # pack.loss["commitment_loss"] = 50 * pack.output.commitment_loss
        pack.metric = pack.metric_fn_t(**pack)  # in autocast may cause inf

        pack.optimiz.zero_grad()
        pack.optimiz.gscale.scale(sum(pack.loss.values())).backward()
        if pack.optimiz.gclip is not None:
            pack.optimiz.gscale.unscale_(pack.optimiz)
            pack.optimiz.gclip(pack.model.parameters())
        pack.optimiz.gscale.step(pack.optimiz)
        pack.optimiz.gscale.update()

        [_.after_step(**pack) for _ in pack.callback_t]

        pack.step_count += 1

        if args.dev: break

    [_.after_epoch(**pack) for _ in pack.callback_t]

    # batches per second（每秒批次数）
    print(f"b/s: {len(pack.dataset_t)} with {len(pack.dataset_t) / (time.time() - t0)}")


@pt.no_grad()
def val_epoch(args,pack):
    pack.model.eval()
    [_.before_epoch(**pack) for _ in pack.callback_v]

    for batch in pack.dataset_v:
        pack.batch = {k: v.cuda() for k, v in batch.items()}

        [_.before_step(**pack) for _ in pack.callback_v]

        with pt.autocast("cuda", enabled=True):
            if args.use_dp:
                pack.output = pack.model(**pack.batch)
            else:
                pack.output = pack.model(pack.batch)
            [_.after_forward(**pack) for _ in pack.callback_v]
            pack.loss = pack.loss_fn(**pack)
            if args.use_dp:
                pack.loss = {k: v.mean() for k,v in pack.loss.items()}
        pack.metric = pack.metric_fn_v(**pack)  # in autocast may cause inf

        [_.after_step(**pack) for _ in pack.callback_v]

        if args.dev: break

    [_.after_epoch(**pack) for _ in pack.callback_v]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    pt.manual_seed(seed)


def main(args):
    pack = Config({})
    print(args)

    cfg_file = Path(args.cfg_file)
    data_path = Path(args.data_dir)
    ckpt_file = Path(args.ckpt_file) if args.ckpt_file else None

    assert cfg_file.name.endswith(".py")
    print("cfg_file: ", cfg_file)
    assert cfg_file.is_file()
    cfg_name = cfg_file.name.split(".")[0]
    cfg = Config.fromfile(args.cfg_file)

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") # 20250614_153045
    if args.dev:
        ts += "_dev"
    save_path = Path(args.save_dir) / cfg_name / f"{args.seed}_{ts}"
    save_path.mkdir(parents=True, exist_ok=True)
    if not args.dev:
        shutil.copy(args.cfg_file, save_path)

    set_seed(args.seed)  # for reproducibility
    pt.backends.cudnn.benchmark = False  # XXX True: faster but stochastic
    pt.backends.cudnn.deterministic = True  # for cuda devices
    pt.use_deterministic_algorithms(True, warn_only=True)  # for all devices

    ## datum init

    work_init_fn = lambda _: set_seed(args.seed)  # for reproducibility
    rng = pt.Generator()
    rng.manual_seed(args.seed)

    cfg.dataset_t.base_dir = cfg.dataset_v.base_dir = data_path
    print("cfg.dataset_t: ", cfg.dataset_t)
    dataset_t = build_from_config(cfg.dataset_t)
    print("data_path:", dataset_t)
    dataload_t = DataLoader(
        dataset_t,
        cfg.batch_size_t,  # TODO XXX TODO XXX TODO XXX TODO XXX // 2
        shuffle=True,
        num_workers=cfg.num_work,
        collate_fn=build_from_config(cfg.collate_fn_t),
        pin_memory=True,
        worker_init_fn=work_init_fn,
        generator=rng,
    )
    dataset_v = build_from_config(cfg.dataset_v)
    dataload_v = DataLoader(
        dataset_v,
        cfg.batch_size_v,
        shuffle=False,
        num_workers=cfg.num_work,
        collate_fn=build_from_config(cfg.collate_fn_v),
        pin_memory=True,
        worker_init_fn=work_init_fn,
        generator=rng,
    )

    ## model init
    model = build_from_config(cfg.model,flag=True)
    # print(model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)

    wrap_flag = False
    if ckpt_file:
        from eminorhan.utils import get_available_models as eminorhan_get_available_models
        from eminorhan.utils import load_checkpoint as eminorhan_load_checkpoint
        from eminorhan.load_vitb16_dino import load_backbone_into_model
        eminorhan_available_models = eminorhan_get_available_models()
        if str(ckpt_file) in eminorhan_available_models:
            ckpt_file = str(ckpt_file)
            print(f"Loading model from eminorhan: {ckpt_file}")
            ckpt_path = eminorhan_load_checkpoint(ckpt_file)
            state_dict = pt.load(ckpt_path, map_location="cpu", weights_only=False)
            load_backbone_into_model(model,state_dict)
            # sd = pt.load(state_dict,map_location="cpu", weights_only=False)
            
            # model.load(sd, cfg.ckpt_map)
        else:
            print(f"Loading model from file: {ckpt_file}")
            ckpt = pt.load(ckpt_file, map_location='cpu')
            model = DPInputAdaptor(model) if args.use_dp else model
            model.load_state_dict(ckpt['model'])
            wrap_flag = True
            # model.load(ckpt_file, cfg.ckpt_map)
    if cfg.freez and not wrap_flag:
        model.freez(cfg.freez)

    model = model.cuda()
    # model.compile()
    if args.use_dp:
        if wrap_flag:
            model = DP(model)
        else:
            model = DP(DPInputAdaptor(model))
        print("Using DataParallel")
    # import pdb; pdb.set_trace()
    if args.visualize:
        # import pdb; pdb.set_trace()
        # image = pt.rand(1,3,256,256).cuda()
        from visualize import visualize
        visualize(args,model.module.wrapped.m)
        import sys; sys.exit(0)
    if args.eval_fgari:
        from eval_fgari import eval
        eval(args,model.module.wrapped.m if args.use_dp else model.wrapped.m)
        import sys; sys.exit(0)

    ## learn init
    if cfg.param_groups:
        cfg.optimiz.params = model.group_params(**cfg.param_groups)
    elif args.use_dp:
        cfg.optimiz.params = model.module.parameters()
    else:
        cfg.optimiz.params = model.parameters()
    optimiz = build_from_config(cfg.optimiz)
    optimiz.gscale = build_from_config(cfg.gscale)
    optimiz.gclip = build_from_config(cfg.gclip)

    loss_fn = MetricWrap(**build_from_config(cfg.loss_fn))
    metric_fn_t = MetricWrap(detach=True, **build_from_config(cfg.metric_fn_t))
    metric_fn_v = MetricWrap(detach=True, **build_from_config(cfg.metric_fn_v))
    # loss_fn.compile()  # TODO lose some ops ???
    # metric_fn.compile()

    for cb in cfg.callback_t + cfg.callback_v:
        if cb.type == "AverageLog":
            cb.log_file = f"{save_path}.txt"
        elif cb.type == "SaveModel":
            cb.save_dir = save_path
    callback_t = build_from_config(cfg.callback_t)
    callback_v = build_from_config(cfg.callback_v)

    ## train loop

    pack.dataset_t = dataload_t
    pack.dataset_v = dataload_v
    pack.model = model
    pack.model_raw = model.module if args.use_dp else model
    pack.optimiz = optimiz
    pack.loss_fn = loss_fn
    pack.metric_fn_t = metric_fn_t
    pack.metric_fn_v = metric_fn_v
    pack.callback_t = callback_t
    pack.callback_v = callback_v
    pack.total_step = cfg.total_step
    pack.val_interval = cfg.val_interval

    epoch_count = 0
    epoch_count_v = 0  # number of val epochs done
    pack.step_count = 0
    [_.before_train(**pack) for _ in pack.callback_t]
    # import pdb; pdb.set_trace()

    while pack.step_count < pack.total_step:  #100 000
        pack.epoch = epoch_count
        pt.cuda.empty_cache()
        # import pdb; pdb.set_trace()
        train_epoch(args,pack)    # pack.step_count += num_batches

        # flag1 = pack.step_count >= (epoch_count_v + 1) * pack.val_interval  # step_num >= val_num * val_interval
        # flag2 = pack.step_count >= pack.total_step
        flag1 = flag2 = True
        if flag1 or flag2:
            pt.cuda.empty_cache()
            with pt.inference_mode(True):
                val_epoch(args,pack)  # pack.step_count not changed
            epoch_count_v += 1

        checkpoint = {
            "epoch": epoch_count,
            "step_count": pack.step_count,
            'model': model.module.state_dict() if args.use_dp else model.state_dict(),
        }
        pt.save(checkpoint, save_path / "checkpoint.pth.tar")

        epoch_count += 1  # pack.epoch += 1?
        print(f"epoch_count: {epoch_count}")

    assert pack.step_count == pack.total_step
    [_.after_train(**pack) for _ in pack.callback_t]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,  # TODO XXX
        # default=np.random.randint(2**32),
    )
    parser.add_argument(
        "--cfg_file",
        type=str,
        # default="config-spot/spot_r-coco.py",
        default="config-spot/spot_r_distill-coco.py",
    )
    parser.add_argument(
        "--data_dir", type=str, default="/media/GeneralZ/Storage/Static/datasets"
    )
    parser.add_argument("--save_dir", type=str, default="save")
    parser.add_argument(
        "--ckpt_file",
        type=str,
        # default="archive-initq/initq_dinosaur_r-coco-encode-twoway-regulariz/best.pth",
    )
    parser.add_argument(
        "--use_dp",
        action="store_true",
        help="Use DataParallel for multi-GPU" \
        " training",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Debug mode",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
    )
    parser.add_argument(
        "--visualize_ttt",
        action="store_true",
    )
    parser.add_argument(
        "--eval_fgari",
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # with pt.autograd.detect_anomaly(True):  # detect NaN
    pt._dynamo.config.suppress_errors = True  # TODO XXX one_hot, interplolate
    main(parse_args())

