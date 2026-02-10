import torch
import torch.onnx
from torch.profiler import profile, record_function, ProfilerActivity
from onnx2torch import convert
import onnx
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='Count FLOPs for ONNX model')
parser.add_argument('--model', '-m', type=str, default='model.onnx', help='Path to ONNX model file')
parser.add_argument('--input-size', '-s', type=int, nargs=2, default=[320, 320], help='Input size (width height)')
parser.add_argument('--batch-size', '-b', type=int, default=1, help='Batch size')
parser.add_argument('--channels', '-c', type=int, default=3, help='Number of input channels')
args = parser.parse_args()

# 转换模型
onnx_model = onnx.load(args.model)
torch_model = convert(onnx_model)
# torch_model.eval()

# # 创建输入
dummy_input = torch.randn(args.batch_size, args.channels, args.input_size[0], args.input_size[1])

# 使用PyTorch Profiler
with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
    with_flops=True
) as prof:
    with record_function("model_inference"):
        output = torch_model(dummy_input)

# 打印FLOPs
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

# 获取总 FLOPs - 方法1: 手动累加
total_flops = 0
for event in prof.key_averages():
    if hasattr(event, 'flops'):
        total_flops += event.flops
print(f"\n总FLOPs: {total_flops:,}")

# 方法2: 使用 events() 获取所有事件
events = prof.events()
total_flops_method2 = sum(event.flops for event in events if hasattr(event, 'flops'))
print(f"总FLOPs (方法2): {total_flops_method2:,}")

# from fvcore.nn import FlopCountAnalysis
# flops = FlopCountAnalysis(torch_model, dummy_input).total()
# print(f"FLOPs: {flops/1e9:.2f} G")