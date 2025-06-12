import tensorrt as trt
import pycuda.driver as cuda
import time
import numpy as np
import cv2
import ctypes
import pycuda.autoinit  # 필수, 지우지 말 것

# Load custom plugin lib
ctypes.CDLL("/home/t/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so")

# Load engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open("model.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# Load and preprocess image
img_path = "/mnt/ssd/LATR/data/openlane/images/validation/segment-17065833287841703_2980_000_3000_000_with_camera_labels/155362886644886900.jpg"
img = cv2.imread(img_path)
img = cv2.resize(img, (960, 720))  # 엔진이 요구하는 정확한 사이즈
img = img.astype(np.float32) / 255.0
img = img.transpose(2, 0, 1)[None]  # (HWC) -> (CHW) -> (NCHW)
img = np.ascontiguousarray(img)

# Set input shape
input_idx = engine.get_binding_index("image")
context.set_binding_shape(input_idx, img.shape)

# Prepare buffers
bindings = [None] * engine.num_bindings
host_outputs = {}
device_outputs = {}

for i in range(engine.num_bindings):
    name = engine.get_binding_name(i)
    dtype = trt.nptype(engine.get_binding_dtype(i))
    shape = context.get_binding_shape(i)

    if engine.binding_is_input(i):
        d_input = cuda.mem_alloc(img.nbytes)
        cuda.memcpy_htod(d_input, img)
        bindings[i] = int(d_input)
    else:
        size = np.prod(shape)
        host_buf = np.empty(size, dtype=dtype)
        device_buf = cuda.mem_alloc(host_buf.nbytes)
        bindings[i] = int(device_buf)
        host_outputs[name] = (host_buf, shape, device_buf)

# Warm-up
for _ in range(10):
    context.execute_v2(bindings)

# Inference 시간 측정
start_time = time.perf_counter()
context.execute_v2(bindings)
cuda.Context.synchronize()
end_time = time.perf_counter()

inference_time_ms = (end_time - start_time) * 1000
print(f"Inference time: {inference_time_ms:.2f} ms")

# 복사 및 출력
for name, (host_buf, shape, device_buf) in host_outputs.items():
    cuda.memcpy_dtoh(host_buf, device_buf)
    output = host_buf.reshape(shape)
    print(f"\n[Output: {name}] Shape: {shape}")
    print("Values:", output)
