import argparse
from mmengine.config import Config, DictAction

from utils.utils import *
from experiments.ddp import *
from experiments.runner import *


def get_args():
    parser = argparse.ArgumentParser()
    # DDP setting
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--use_slurm', default=False, action='store_true')

    parser.add_argument('--export_onnx', action='store_true', help='Export model to ONNX format')

    # exp setting
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='overwrite config param.')
    return parser.parse_args()


def main():
    args = get_args()
    # define runner to begin training or evaluation
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # initialize distributed data parallel set
    if args.distributed:
        ddp_init(args)
    cfg.merge_from_dict(vars(args))
    
    runner = Runner(cfg)
    if not cfg.evaluate:
        runner.train()
    else:
        runner.eval()

if __name__ == '__main__':
    main()