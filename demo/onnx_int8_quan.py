import os
import argparse
import onnx
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantType,
    QuantFormat, QDQQuantizer
)

class RandomDataReader(CalibrationDataReader):
    def __init__(self, input_name, input_shape, num_samples=10, input_range=(0, 1)):
        self.input_name = input_name
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.current_sample = 0
        self.input_range = input_range
        self.enum_data = None
        
    def get_next(self):
        if self.current_sample < self.num_samples:
            self.current_sample += 1
            # 生成随机输入数据，范围与实际输入相似
            input_data = np.random.uniform(
                self.input_range[0], self.input_range[1], 
                size=self.input_shape
            ).astype(np.float32)
            return {self.input_name: input_data}
        return None

class RealDataReader(CalibrationDataReader):
    def __init__(self, input_name, data_dir):
        self.input_name = input_name
        self.data_dir = data_dir
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.current_idx = 0
        
    def get_next(self):
        if self.current_idx < len(self.data_files):
            file_path = os.path.join(self.data_dir, self.data_files[self.current_idx])
            self.current_idx += 1
            input_data = np.load(file_path).astype(np.float32)
            return {self.input_name: input_data}
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="ONNX FP32 to INT8 Quantization Tool")
    parser.add_argument('--input', type=str, required=True, help='Path to input FP32 ONNX model')
    parser.add_argument('--output', type=str, default='model_int8.onnx', help='Path to output INT8 ONNX model')
    parser.add_argument('--calibration', type=str, choices=['random', 'real'], default='random', help='Calibration data type')
    parser.add_argument('--calibration-dir', type=str, default=None, help='Directory containing real calibration data (.npy files)')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of calibration samples')
    parser.add_argument('--input-range', type=float, nargs=2, default=(-2.13, 2.65), help='Input data range for random calibration')
    parser.add_argument('--weight-type', type=str, choices=['quint8', 'qint8'], default='qint8', help='Weight quantization type')
    parser.add_argument('--activation-type', type=str, choices=['quint8', 'qint8'], default='qint8', help='Activation quantization type')
    parser.add_argument('--quant-format', type=str, choices=['QDQ', 'QOP'], default='QDQ', help='Quantization format')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 设置文件路径
    fp32_model_path = args.input
    int8_model_path = args.output
    
    # 2. 加载并检查FP32模型
    if not os.path.exists(fp32_model_path):
        print(f"Error: FP32 model file '{fp32_model_path}' not found!")
        return
    
    print(f"Loading FP32 model from {fp32_model_path}...")
    try:
        fp32_model = onnx.load(fp32_model_path)
        onnx.checker.check_model(fp32_model)
        print("FP32 model loaded and validated successfully.")
    except Exception as e:
        print(f"Error loading or validating model: {e}")
        return
    
    # 3. 获取模型输入信息
    try:
        input_tensor = fp32_model.graph.input[0]
        input_name = input_tensor.name
        input_shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                input_shape.append(dim.dim_value)
            else:
                # 如果是动态维度，使用默认值1
                input_shape.append(1)
        
        print(f"Input name: {input_name}")
        print(f"Input shape: {input_shape}")
    except Exception as e:
        print(f"Error getting model input info: {e}")
        return
    
    # 4. 创建校准数据读取器
    try:
        if args.calibration == 'real':
            if not args.calibration_dir or not os.path.exists(args.calibration_dir):
                print("Error: Real calibration data directory not provided or does not exist!")
                return
            calibration_reader = RealDataReader(input_name, args.calibration_dir)
            print(f"Using real calibration data from {args.calibration_dir}")
        else:
            calibration_reader = RandomDataReader(
                input_name, input_shape, 
                num_samples=args.num_samples, 
                input_range=args.input_range
            )
            print(f"Using random calibration data with range {args.input_range}")
    except Exception as e:
        print(f"Error creating calibration reader: {e}")
        return
    
    # 5. 设置量化参数
    weight_type = QuantType.QUInt8 if args.weight_type == 'quint8' else QuantType.QInt8
    activation_type = QuantType.QUInt8 if args.activation_type == 'quint8' else QuantType.QInt8
    quant_format = QuantFormat.QDQ if args.quant_format == 'QDQ' else QuantFormat.QOperator
    
    # 6. 执行静态量化
    print("Starting INT8 quantization...")
    try:
        # 移除不支持的参数，调整为兼容旧版本ONNX Runtime的参数
        quantize_static(
            model_input=fp32_model_path,
            model_output=int8_model_path,
            calibration_data_reader=calibration_reader,
            weight_type=weight_type,
            activation_type=activation_type,
            quant_format=quant_format,
            # optimize_model=True,
            per_channel=False,  # 可以根据需要设置为True
            reduce_range=True   # 可以根据需要设置为False
        )
        print(f"INT8 quantization completed. Quantized model saved to {int8_model_path}.")
    except Exception as e:
        print(f"Error during quantization: {e}")
        return
    
    # 7. 创建测试输入数据
    print("Creating test input data...")
    test_input = np.random.uniform(args.input_range[0], args.input_range[1], size=input_shape).astype(np.float32)
    
    # 8. 运行FP32模型推理
    print("Running FP32 model inference...")
    try:
        fp32_session = ort.InferenceSession(fp32_model_path)
        fp32_outputs = fp32_session.run(None, {input_name: test_input})
    except Exception as e:
        print(f"Error running FP32 model inference: {e}")
        return
    
    # 9. 运行INT8模型推理
    print("Running INT8 model inference...")
    try:
        int8_session = ort.InferenceSession(int8_model_path)
        int8_outputs = int8_session.run(None, {input_name: test_input})
    except Exception as e:
        print(f"Error running INT8 model inference: {e}")
        return
    
    # 10. 比较输出结果
    print("Comparing outputs...")
    try:
        for i, (fp32_out, int8_out) in enumerate(zip(fp32_outputs, int8_outputs)):
            print(f"\nOutput {i+1}:")
            print(f"  FP32 shape: {fp32_out.shape}")
            print(f"  INT8 shape: {int8_out.shape}")
            
            # 计算误差指标
            abs_error = np.abs(fp32_out - int8_out)
            mean_abs_error = np.mean(abs_error)
            max_abs_error = np.max(abs_error)
            
            # 计算相对误差（避免除以零）
            epsilon = 1e-8
            rel_error = abs_error / (np.abs(fp32_out) + epsilon)
            mean_rel_error = np.mean(rel_error)
            
            print(f"  Mean absolute error: {mean_abs_error:.6f}")
            print(f"  Max absolute error: {max_abs_error:.6f}")
            print(f"  Mean relative error: {mean_rel_error:.6%}")
            
            # 计算均方误差
            mse = np.mean((fp32_out - int8_out) ** 2)
            print(f"  Mean squared error: {mse:.6f}")
            
            # 计算余弦相似度
            cos_sim = np.dot(fp32_out.flatten(), int8_out.flatten()) / (
                np.linalg.norm(fp32_out) * np.linalg.norm(int8_out) + epsilon
            )
            print(f"  Cosine similarity: {cos_sim:.6f}")
            
            # 打印部分输出值进行直观比较
            print("  Sample FP32 values:", fp32_out.flatten()[:5])
            print("  Sample INT8 values:", int8_out.flatten()[:5])
    except Exception as e:
        print(f"Error comparing outputs: {e}")
        return
    
    print("\nQuantization validation completed successfully!")
    print(f"\nQuantization Summary:")
    print(f"  Input model: {fp32_model_path}")
    print(f"  Output model: {int8_model_path}")
    print(f"  Calibration type: {args.calibration}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Weight type: {args.weight_type}")
    print(f"  Activation type: {args.activation_type}")
    print(f"  Quantization format: {args.quant_format}")

if __name__ == "__main__":
    main()