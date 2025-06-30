import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import ctypes
import json
import cv2
from tqdm import tqdm
import torch
import copy

import argparse
from mmengine.config import Config, DictAction
from data.Load_Data import *
from data.apollo_dataset import ApolloLaneDataset

from experiments.gpu_utils import is_main_process
from utils.utils import *
from utils import eval_3D_lane, eval_3D_once
from utils import eval_3D_lane_apollo

from models.latr import LATR
import pickle

prev_cls = None

def infer_single(engine, context, sample, cuda_stream=None):
    bindings = [None] * engine.num_bindings
    host_outputs = {}

    # Step 1: Set all input shapes first
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            name = engine.get_binding_name(i)
            tensor = sample[name]
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().numpy()
            elif not isinstance(tensor, np.ndarray):
                raise ValueError(f"Unexpected input '{name}': {type(tensor)}")      
            # context.set_binding_shape(i, tensor.shape)  # 반드시 먼저 호출

    # Step 2: Allocate memory and bind
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        shape = context.get_binding_shape(i)
        size = int(np.prod(shape))

        if engine.binding_is_input(i):
            tensor = sample[name]
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().numpy()
            bindings[0] = cuda.mem_alloc(tensor.nbytes)
            cuda.memcpy_htod(bindings[0], tensor)
            bindings[i] = int(bindings[0])
        else:
            host_buf = np.empty(size, dtype=dtype)
            device_buf = cuda.mem_alloc(host_buf.nbytes)
            bindings[i] = int(device_buf)
            host_outputs[name] = (host_buf, shape, device_buf)

    # Step 3: Execute
    if not context.all_binding_shapes_specified:
        raise RuntimeError("Not all binding shapes specified.")
    context.execute_v2(bindings)
    cuda.Context.synchronize()

    # Step 4: Retrieve outputs
    result_dict = {}
    for name, (host_buf, shape, device_buf) in host_outputs.items():
        cuda.memcpy_dtoh(host_buf, device_buf)
        result_dict[name] = torch.from_numpy(host_buf.reshape(shape))

    return result_dict

def infer_singlev2(
        engine, context, sample, bindings, h_input, d_input, h_outputs,
        cuda_stream=None, targets=["all_line_preds", "all_cls_scores"]
    ):
    """
    name 0 image
    name 1 out_featList
    name 2 first_neck_out
    name 3 neck_out
    name 4 onnx::Shape_944
    name 5 input.424
    name 6 input.428
    name 7 input.432
    name 8 onnx::Shape_1032
    name 9 all_cls_scores
    name 10 all_line_preds
    """
    # print(f"mean: {sample['image'].mean()}, std: {sample['image'].std()}")
    image = sample["image"].detach().cpu().numpy()
    # image = np.random.rand(1, 3, 720, 960)
    np.copyto(h_input, image.ravel())
    cuda.memcpy_htod_async(d_input, h_input, cuda_stream)

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        if not engine.binding_is_input(i):
            if not name in targets:
                h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(i)), dtype=np.float32)
                bindings[i] = cuda.mem_alloc(h_output.nbytes)
            else:
                bindings[i] = cuda.mem_alloc(h_outputs[name].nbytes)

    context.execute_async_v2(bindings=[d_input]+[int(b) for b in bindings[1:]], stream_handle=cuda_stream.handle)
    
    shapes = {k: None for k in targets}
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        if name in targets:
            shapes[name] = engine.get_binding_shape(i)
            cuda.memcpy_dtoh_async(h_outputs[name], bindings[i], cuda_stream)
    
    cuda_stream.synchronize()
    
    results_dict = {k: None for k in targets}
    for name in h_outputs.keys():
        results_dict[name] = torch.from_numpy(h_outputs[name].reshape(shapes[name]))
        # print(f"{name} mean: {results_dict[name].mean()}, std: {results_dict[name].std()}")
    # TODO: get outputs
    # cuda.memcpy_dtoh_async(h_output, d_output, cuda_stream)

    # cls_raw = outputs["all_cls_scores"][-1]     # (2, 1, 40, 21)
    # coord_raw = outputs["all_line_preds"][-1]     # (2, 1, 40, 60)    

    return results_dict

def trt_eval_loop(
        engine, 
        context, 
        dataloader, 
        args, 
        infer_func, 
        logger, 
        evaluator, 
        width=720, 
        height=960,
        targets=["all_cls_scores", "all_line_preds"]
    ):
    global prev_cls
    pred_lines_sub = []
    gt_lines_sub = []

    cuda_stream = cuda.Stream()

    h_input = cuda.pagelocked_empty(trt.volume((1, 3, width, height)), dtype=np.float32)
    bindings = [None] * engine.num_bindings
    d_input = cuda.mem_alloc(h_input.nbytes)

    for i, sample in enumerate(tqdm(dataloader, ncols=50)):

        json_files = sample.pop('idx_json_file')

        h_outputs = {
            "all_cls_scores": cuda.pagelocked_empty(trt.volume((2, 1, 40, 21)), dtype=np.float32),
            "all_line_preds": cuda.pagelocked_empty(trt.volume((2, 1, 40, 60)), dtype=np.float32),
        }

        output = infer_func(engine, context, sample, bindings, h_input, d_input, h_outputs, cuda_stream)

        # --- ONNX 출력 후처리
        all_line_preds = output['all_line_preds'] # in ground coordinate system
        all_cls_scores = output['all_cls_scores']

        all_line_preds = all_line_preds[-1]
        all_cls_scores = all_cls_scores[-1]
        num_el = all_cls_scores.shape[0]

        # --- 메타 정보 처리
        cam_extrinsics_all = sample.get('cam_extrinsics', None)
        cam_intrinsics_all = sample.get('cam_intrinsics', None)

        for j in range(num_el):

            json_file = json_files[j]
            if cam_extrinsics_all is not None:
                extrinsic = cam_extrinsics_all[j].cpu().numpy()
                intrinsic = cam_intrinsics_all[j].cpu().numpy()

            with open(json_file, 'r') as f:
                if 'apollo' in args.dataset_name:
                    json_line = json.loads(f.read())
                    if 'extrinsic' not in json_line and cam_extrinsics_all is not None:
                        json_line['extrinsic'] = extrinsic
                    if 'intrinsic' not in json_line and cam_intrinsics_all is not None:
                        json_line['intrinsic'] = intrinsic
                else:
                        file_lines = [line for line in f]
                        json_line = json.loads(file_lines[0])

                json_line['json_file'] = json_file

                if 'once' in args.dataset_name:
                    json_line["file_path"] = json_file.replace('val', 'data').replace('.json', '.jpg')

                gt_lines_sub.append(copy.deepcopy(json_line))

                # --- 예측 결과 후처리
                lane_pred = all_line_preds[j].cpu().numpy()
                cls_pred = torch.argmax(all_cls_scores[j], dim=-1).detach().cpu().numpy()
                pos_lanes = lane_pred[cls_pred > 0]

                if args.num_category > 1:
                    scores_pred = torch.softmax(all_cls_scores[j][cls_pred > 0], dim=-1).detach().cpu().numpy()
                else:
                    scores_pred = torch.sigmoid(all_cls_scores[j][cls_pred > 0]).detach().cpu().numpy()
                # print("max score:", scores_pred.max())
                # print("raw cls scores:\n", all_cls_scores[:5])

                if pos_lanes.shape[0]:
                    lanelines_pred = []
                    lanelines_prob = []
                    xs = pos_lanes[:, 0:args.num_y_steps]
                    ys = np.tile(np.array(args.anchor_y_steps).copy()[None, :], (xs.shape[0], 1))
                    zs = pos_lanes[:, args.num_y_steps:2*args.num_y_steps]
                    vis = pos_lanes[:, 2*args.num_y_steps:]

                    for idx in range(pos_lanes.shape[0]):
                        cur_vis = vis[idx] > 0
                        if cur_vis.sum() < 2:
                            continue
                        cur_xs = xs[idx][cur_vis]
                        cur_ys = ys[idx][cur_vis]
                        cur_zs = zs[idx][cur_vis]

                        lanelines_pred.append([])
                        for tmp_inner_idx in range(cur_xs.shape[0]):
                            lanelines_pred[-1].append(
                                [cur_xs[tmp_inner_idx],
                                    cur_ys[tmp_inner_idx],
                                    cur_zs[tmp_inner_idx]])
                        lanelines_prob.append(scores_pred[idx].tolist())
                else:
                    lanelines_pred, lanelines_prob = [], []

                json_line["pred_laneLines"] = lanelines_pred
                json_line["pred_laneLines_prob"] = lanelines_prob
                pred_lines_sub.append(copy.deepcopy(json_line))
                
                if args.save_pred:
                    ROOT_DIR  = Path(__file__).resolve().parent
                    if not args.evaluate_case: # training or validation case
                        #TODO : change this directory
                        
                        save_dir  = ROOT_DIR / "openlane_pred_json_trt"
                        os.makedirs(save_dir, exist_ok=True)
                        output_path = os.path.join(save_dir, f"Opred_{i}_{j}.pickle")
                        with open(output_path, 'wb') as f:
                            pickle.dump(json_line, f)

                    else:
                        save_dir = ROOT_DIR / "TCAR_pred_json"
                        os.makedirs(save_dir, exist_ok=True)
                        # 파일 경로 # only batch sise 1
                        output_path = os.path.join(save_dir, f"Tpred_{i}.pickle")
                        with open(output_path, 'wb') as f:
                            pickle.dump(json_line, f)

    # --- 평가 호출
    if 'openlane' in args.dataset_name:
        eval_stats = evaluator.bench_one_submit_ddp(
            pred_lines_sub, gt_lines_sub, args.model_name,
            args.pos_threshold, vis=False)
    elif 'once' in args.dataset_name:
        eval_stats = evaluator.lane_evaluation(
            args.data_dir + 'val', '%s/once_pred/test' % args.save_path,
            args.eval_config_dir, args)
    elif 'apollo' in args.dataset_name:
        logger.info(' >>> eval mAP | [0.05, 0.95]')
        eval_stats = evaluator.bench_one_submit_ddp(
            pred_lines_sub, gt_lines_sub, args.model_name,
            args.pos_threshold, vis=False)
    else:
        raise NotImplementedError

    return [], eval_stats

def log_eval_stats(eval_stats):

    if is_main_process():
        logger.info("===> Evaluation laneline F-measure: {:.8f}".format(eval_stats[0]))
        logger.info("===> Evaluation laneline Recall: {:.8f}".format(eval_stats[1]))
        logger.info("===> Evaluation laneline Precision: {:.8f}".format(eval_stats[2]))
        logger.info("===> Evaluation laneline Category Accuracy: {:.8f}".format(eval_stats[3]))
        logger.info("===> Evaluation laneline x error (close): {:.8f} m".format(eval_stats[4]))
        logger.info("===> Evaluation laneline x error (far): {:.8f} m".format(eval_stats[5]))
        logger.info("===> Evaluation laneline z error (close): {:.8f} m".format(eval_stats[6]))
        logger.info("===> Evaluation laneline z error (far): {:.8f} m".format(eval_stats[7]))
        

def get_valid_dataset(arg):
    args = arg
    
    if 'openlane' in args.dataset_name:
        if not args.evaluate_case:
            valid_dataset = LaneDataset(args.dataset_dir, args.data_dir + 'validation/', args)
        else:
            # TODO eval case
            valid_dataset = LaneDataset(args.dataset_dir, args.data_dir + 'test/up_down_case/', args)

    elif 'once' in args.dataset_name:
        valid_dataset = LaneDataset(args.dataset_dir, ops.join(args.data_dir, 'val/'), args)
    else:
        valid_dataset = ApolloLaneDataset(args.dataset_dir, os.path.join(args.data_dir, 'test.json'), args)

    valid_loader, valid_sampler = get_loader(valid_dataset, args)
    return valid_dataset, valid_loader, valid_sampler

def get_args():
    parser = argparse.ArgumentParser()
    # DDP setting
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--use_slurm', default=False, action='store_true')
    parser.add_argument('--save_pred', default=False, action='store_true')

    parser.add_argument('--export_onnx', action='store_true', help='Export model to ONNX format')

    # exp setting
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='overwrite config param.')
    return parser.parse_args()

if __name__ == '__main__':
    # Load plugin
    ctypes.CDLL("/home/t/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so")
    ctypes.CDLL("/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so", ctypes.RTLD_GLOBAL)
    ctypes.CDLL("/usr/lib/aarch64-linux-gnu/libnvonnxparser.so", ctypes.RTLD_GLOBAL)

    # Load engine
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    with open("./tensorrt/model_0620_fp32.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    

    # Binding index
    input_name = "image"
    input_idx = engine.get_binding_index(input_name)

    args = get_args()
    # define runner to begin training or evaluation
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg.merge_from_dict(vars(args))

    cfg.distributed = False

    logger = create_logger(cfg)
    valid_dataset, valid_loader, valid_sampler = get_valid_dataset(cfg)

    if 'openlane' in cfg.dataset_name:
        evaluator = eval_3D_lane.LaneEval(cfg, logger=logger)
    elif 'apollo' in cfg.dataset_name:
        evaluator = eval_3D_lane_apollo.LaneEval(cfg, logger=logger)
    elif 'once' in cfg.dataset_name:
        evaluator = eval_3D_once.LaneEval()
    else:
        assert False

    # TRT 추론 및 평가 수행
    _, eval_stats = trt_eval_loop(
        engine=engine,
        context=context,
        dataloader=valid_loader,
        args=cfg,
        infer_func=infer_singlev2,
        logger=logger,
        evaluator=evaluator
    )

    # 평가 결과 출력
    log_eval_stats(eval_stats)