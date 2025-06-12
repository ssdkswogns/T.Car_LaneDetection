import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes

onnx_file = "./tensorrt/model_0609_partial.onnx"
engine_file = "./tensorrt/model_0609_partial.engine"
plugin_path = "/home/t/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so"

ctypes.CDLL(plugin_path)

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

with open(onnx_file, 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parsing failed")

# 고정 input shape 설정 확인
input_tensor = network.get_input(0)
print(input_tensor.shape)
if -1 in input_tensor.shape:
    input_tensor.shape = (1, 3, 720, 960)

config = builder.create_builder_config()
config.max_workspace_size = 8192 * (1 << 20)  # 8GB for Jetson
config.set_flag(trt.BuilderFlag.FP16)
# config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)

print("🔧 Building engine...")
engine = builder.build_serialized_network(network, config)
with open(engine_file, 'wb') as f:
    f.write(engine)

print(f"✅ Engine saved to {engine_file}")

