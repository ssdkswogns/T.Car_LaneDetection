import pycuda.autoinit

import tensorrt as trt
import pycuda.driver as cuda

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
from models.latr_head import LATRHead

import gc

# Init CUDA context explicitly for Jetson (no autoinit)
cuda.init()
device = cuda.Device(0)
context = device.make_context()

class TensorRTInference:
    def __init__(self, engine_path):
        # Load plugin (optional)
        try:
            ctypes.CDLL("/home/t/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so")
        except Exception as e:
            print(f"Warning: Failed to load TensorRT plugin: {e}")

        self.logger = trt.Logger(trt.Logger.WARNING)
        try:
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                engine_data = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_data)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}: {e}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.input_idx = None
        self.output_idx = None
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            if self.engine.binding_is_input(i):
                self.input_idx = i
                self.input_name = name
            else:
                self.output_idx = i
                self.output_name = name

        self.d_input = None
        self.d_output = None
        self.h_output = None
        self.input_size = 0
        self.output_size = 0

    def __del__(self):
        try:
            if hasattr(self, 'd_input') and self.d_input:
                self.d_input.free()
            if hasattr(self, 'd_output') and self.d_output:
                self.d_output.free()
        except Exception as e:
            print(f"[WARNING] Memory free error: {e}")

    def infer(self, image_tensor):
        try:
            gc.collect()
            torch.cuda.empty_cache()

            if isinstance(image_tensor, torch.Tensor):
                image_np = image_tensor.detach().cpu().numpy()
            else:
                image_np = image_tensor

            image_np = np.ascontiguousarray(image_np, dtype=np.float32)
            input_shape = image_np.shape

            self.context.set_input_shape(self.input_name, input_shape)
            output_shape = self.context.get_tensor_shape(self.output_name)
            if any(dim <= 0 for dim in output_shape):
                raise RuntimeError(f"Invalid output shape: {output_shape}")

            output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

            input_size_needed = image_np.nbytes

            if self.d_input is None or getattr(self, 'input_shape', None) != input_shape:
                if self.d_input:
                    self.d_input.free()
                self.d_input = cuda.mem_alloc(input_size_needed)
                self.input_size = input_size_needed
                self.input_shape = input_shape

            expected_output_bytes = (int(np.prod(output_shape)) * np.dtype(output_dtype).itemsize + 32)

            if self.h_output is None or self.h_output.shape != output_shape:
                self.h_output = np.empty(output_shape, dtype=output_dtype)

                if self.d_output:
                    self.d_output.free()
                self.d_output = cuda.mem_alloc(expected_output_bytes)
                self.output_size = expected_output_bytes

            # if self.d_output is None or self.output_size < output_size_needed:
            #     if self.d_output: self.d_output.free()
            #     self.d_output = cuda.mem_alloc(output_size_needed)
            #     self.output_size = output_size_needed

            # self.h_output = np.empty(output_shape, dtype=output_dtype)

            cuda.memcpy_htod(self.d_input, image_np)

            bindings = {
                self.input_name: int(self.d_input),
                self.output_name: int(self.d_output)
            }
            success = self.context.execute_v2(list(bindings.values()))

            if not success:
                raise RuntimeError("TensorRT execute_v2 returned False")

            cuda.Context.synchronize()
            cuda.memcpy_dtoh(self.h_output, self.d_output)

            return {"neck_out": self.h_output.copy()}

        except Exception as e:
            raise RuntimeError(f"TensorRT inference failed: {str(e)}")


def postprocess_outputs(outputs, args):
    """Extract classification scores and coordinates from model outputs"""
    cls_raw = outputs["all_cls_scores"][-1]     # (1, 40, 21)
    coord_raw = outputs["all_line_preds"][-1]   # (1, 40, 60)
    
    cls_scores = cls_raw[0]     # (40, 21)
    coords = coord_raw[0]       # (40, 60)
    
    return cls_scores, coords

def trt_eval_loop(trt_model, head, dataloader, args, logger, evaluator):
    """Main evaluation loop using TensorRT inference"""
    pred_lines_sub = []
    gt_lines_sub = []
    
    print(f"Starting evaluation with {len(dataloader)} samples...")
    
    for i, sample in enumerate(tqdm(dataloader, desc="Evaluating", ncols=80)):
        try:
            json_files = sample.pop('idx_json_file')
            
            # TensorRT inference (backbone + neck)
            image_input = sample['image']
            neck_outputs = trt_model.infer(image_input)
            
            # Convert neck output to torch tensor
            neck_out = torch.from_numpy(neck_outputs["neck_out"]).to(torch.float32).to(device)
            
            # Head inference
            head_inputs = {
                'x': neck_out,
                'lane_idx': sample['seg_idx_label'].to(device),
                'seg': sample['seg_label'].to(device),
                'lidar2img': sample['lidar2img'].to(device),
                'pad_shape': sample['pad_shape'].to(device),
                'ground_lanes': None,
                'ground_lanes_dense': None,
                'image': sample['image'].to(device)
            }

            with torch.no_grad():
                outputs_dict = head(head_inputs, is_training=False)
            
            # Post-process outputs
            all_cls_scores, all_line_preds = postprocess_outputs(outputs_dict, args)

            # print("mean:", all_cls_scores.mean().item(), "std:", all_cls_scores.std().item())
            
            # Process metadata
            cam_extrinsics_all = sample.get('cam_extrinsics', None)
            cam_intrinsics_all = sample.get('cam_intrinsics', None)
            json_file = json_files[0]  # batch size 1
            
            # Load ground truth
            with open(json_file, 'r') as f:
                if 'apollo' in args.dataset_name:
                    json_line = json.loads(f.read())
                    if 'extrinsic' not in json_line and cam_extrinsics_all is not None:
                        json_line['extrinsic'] = cam_extrinsics_all[0].cpu().numpy()
                    if 'intrinsic' not in json_line and cam_intrinsics_all is not None:
                        json_line['intrinsic'] = cam_intrinsics_all[0].cpu().numpy()
                else:
                    json_line = json.loads(f.readline())
                
                json_line['json_file'] = json_file
                
                if 'once' in args.dataset_name:
                    json_line["file_path"] = json_file.replace('val', 'data').replace('.json', '.jpg')
                
                gt_lines_sub.append(copy.deepcopy(json_line))
            
            # Process predictions
            cls_pred = torch.argmax(all_cls_scores, dim=-1).detach().cpu().numpy()
            pos_lanes = all_line_preds[cls_pred > 0].detach().cpu().numpy()
            
            if args.num_category > 1:
                scores_pred = torch.softmax(all_cls_scores[cls_pred > 0], dim=-1).detach().cpu().numpy()
            else:
                scores_pred = torch.sigmoid(all_cls_scores[cls_pred > 0]).detach().cpu().numpy()
            
            # Extract lane lines
            lanelines_pred = []
            lanelines_prob = []
            
            if pos_lanes.shape[0] > 0:
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
                    
                    points = [[float(cur_xs[k]), float(cur_ys[k]), float(cur_zs[k])]
                             for k in range(len(cur_xs))]
                    lanelines_pred.append(points)
                    lanelines_prob.append(scores_pred[idx].tolist())
            
            json_line["pred_laneLines"] = lanelines_pred
            json_line["pred_laneLines_prob"] = lanelines_prob
            pred_lines_sub.append(copy.deepcopy(json_line))
            
        except Exception as e:
            logger.error(f"Error processing sample {i}: {str(e)}")
            continue
    
    # Evaluation
    try:
        if 'openlane' in args.dataset_name:
            eval_stats = evaluator.bench_one_submit_ddp(
                pred_lines_sub, gt_lines_sub, args.model_name,
                args.pos_threshold, vis=False)
        elif 'once' in args.dataset_name:
            eval_stats = evaluator.lane_evaluation(
                args.data_dir + 'val', f'{args.save_path}/once_pred/test',
                args.eval_config_dir, args)
        elif 'apollo' in args.dataset_name:
            logger.info(' >>> eval mAP | [0.05, 0.95]')
            eval_stats = evaluator.bench_one_submit_ddp(
                pred_lines_sub, gt_lines_sub, args.model_name,
                args.pos_threshold, vis=False)
        else:
            raise NotImplementedError(f"Evaluation not implemented for dataset: {args.dataset_name}")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        eval_stats = [0.0] * 8  # Default stats
    
    return pred_lines_sub, eval_stats

def log_eval_stats(eval_stats, logger):
    """Log evaluation statistics"""
    if is_main_process() and len(eval_stats) >= 8:
        logger.info("===> Evaluation Results:")
        logger.info(f"F-measure: {eval_stats[0]:.8f}")
        logger.info(f"Recall: {eval_stats[1]:.8f}")
        logger.info(f"Precision: {eval_stats[2]:.8f}")
        logger.info(f"Category Accuracy: {eval_stats[3]:.8f}")
        logger.info(f"X error (close): {eval_stats[4]:.8f} m")
        logger.info(f"X error (far): {eval_stats[5]:.8f} m")
        logger.info(f"Z error (close): {eval_stats[6]:.8f} m")
        logger.info(f"Z error (far): {eval_stats[7]:.8f} m")

def get_valid_dataset(args):
    """Get validation dataset and dataloader"""
    if 'openlane' in args.dataset_name:
        if not args.evaluate_case:
            valid_dataset = LaneDataset(args.dataset_dir, args.data_dir + 'validation/', args)
        else:
            valid_dataset = LaneDataset(args.dataset_dir, args.data_dir + 'test/up_down_case/', args)
    elif 'once' in args.dataset_name:
        valid_dataset = LaneDataset(args.dataset_dir, os.path.join(args.data_dir, 'val/'), args)
    else:
        valid_dataset = ApolloLaneDataset(args.dataset_dir, os.path.join(args.data_dir, 'test.json'), args)
    
    valid_loader, valid_sampler = get_loader(valid_dataset, args)
    return valid_dataset, valid_loader, valid_sampler

def get_args():
    """Parse command line arguments"""
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
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='overwrite config param.')
    parser.add_argument('--engine-path', type=str, default="./tensorrt/model_0609_partial.engine", 
                       help='Path to TensorRT engine file')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    # Load configuration
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.merge_from_dict(vars(args))
    cfg.distributed = False
    
    # Initialize logger
    logger = create_logger(cfg)
    logger.info("Starting TensorRT inference evaluation...")
    
    try:
        # Load dataset
        valid_dataset, valid_loader, valid_sampler = get_valid_dataset(cfg)
        logger.info(f"Dataset loaded: {len(valid_dataset)} samples")
        
        # Initialize evaluator
        if 'openlane' in cfg.dataset_name:
            evaluator = eval_3D_lane.LaneEval(cfg, logger=logger)
        elif 'apollo' in cfg.dataset_name:
            evaluator = eval_3D_lane_apollo.LaneEval(cfg, logger=logger)
        elif 'once' in cfg.dataset_name:
            evaluator = eval_3D_once.LaneEval()
        else:
            raise ValueError(f"Unsupported dataset: {cfg.dataset_name}")
        
        # Initialize LATR head
        model = LATR(cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # head =  LATRHead(
        #         args=args,
        #         dim=_dim_,
        #         num_group=num_group,
        #         num_convs=4,
        #         in_channels=_dim_,
        #         kernel_dim=_dim_,
        #         position_range=args.position_range,
        #         top_view_region=args.top_view_region,
        #         positional_encoding=dict(
        #             type='SinePositionalEncoding',
        #             num_feats=_dim_// 2, normalize=True),
        #         num_query=num_query,
        #         pred_dim= num_y_steps,
        #         num_classes=args.num_category,
        #         embed_dims=_dim_,
        #         transformer=args.transformer,
        #         sparse_ins_decoder=args.sparse_ins_decoder,
        #         **args.latr_cfg.get('head', {}),
        #         trans_params=args.latr_cfg.get('trans_params', {})
        #     ).to(device)

        head = model.head.to(device)
        head.eval()

        # Initialize TensorRT model
        trt_model = TensorRTInference(args.engine_path)
        logger.info(f"TensorRT engine loaded successfully from {args.engine_path}")

        # Run evaluation
        pred_lines, eval_stats = trt_eval_loop(
            trt_model=trt_model,
            head=head,
            dataloader=valid_loader,
            args=cfg,
            logger=logger,
            evaluator=evaluator
        )
        
        # # Log results
        log_eval_stats(eval_stats, logger)
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}")
        raise