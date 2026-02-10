import onnx
from onnxruntime.quantization import quantize_dynamic, quantize_static
from onnxruntime.quantization import QuantType

def convert_fp32_to_fp16(input_model_path, output_model_path):
    """将FP32模型转换为FP16"""
    from onnxconverter_common import float16
    
    # 加载原始模型
    model = onnx.load(input_model_path)
    
    # 转换为FP16
    model_fp16 = float16.convert_float_to_float16(model)
    
    # 保存新模型
    onnx.save(model_fp16, output_model_path)
    print(f"模型已转换为FP16并保存到: {output_model_path}")

# 使用示例
convert_fp32_to_fp16("nanodet_brush.onnx", "nanodet_brush_fp16.onnx")