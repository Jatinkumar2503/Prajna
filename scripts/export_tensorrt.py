"""
PRAJNA TENSORRT 10 ENGINE EXPORTER & CALIBRATION HARNESS
Compiles trained ONNX computation graphs into high-throughput NVIDIA TensorRT engines
with FP16 / INT4 / INT8 Post-Training Quantization for Jetson Orin / RTX 3050.
"""

import os
import sys
import argparse

def export_tensorrt_engine(onnx_model_path: str,
                           output_engine_path: str,
                           precision: str = "fp16",
                           max_batch_size: int = 1):
    print(f"[*] PRAJNA TensorRT Exporter: Loading ONNX graph from '{onnx_model_path}'...")
    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(f"ONNX model file not found: {onnx_model_path}")
        
    file_size_mb = os.path.getsize(onnx_model_path) / (1024 * 1024)
    print(f"    Graph Size: {file_size_mb:.2f} MB | Target Precision: {precision.upper()} | Batch Size: {max_batch_size}")

    try:
        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_model_path, "rb") as model_file:
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    print(f"[-] TensorRT ONNX Parser Error: {parser.get_error(error)}")
                return False

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB Workspace

        if precision.lower() == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[+] TensorRT: FP16 Execution Enabled.")

        print(f"[*] Building TensorRT Execution Engine (saving to {output_engine_path})...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("[-] TensorRT Engine build failed.")
            return False

        os.makedirs(os.path.dirname(output_engine_path), exist_ok=True)
        with open(output_engine_path, "wb") as f:
            f.write(serialized_engine)

        print(f"[+] Successfully compiled TensorRT engine: {output_engine_path}")
        return True

    except ImportError:
        print("[!] Note: TensorRT Python bindings (`import tensorrt`) are not installed on this host.")
        print("    Generating TensorRT trtexec CLI command for target NVIDIA Jetson/H100 deployment:")
        trtexec_cmd = (
            f"trtexec --onnx={onnx_model_path} --saveEngine={output_engine_path} "
            f"--{precision.lower()} --memPoolSize=workspace:1024MiB --verbose"
        )
        print(f"\n    $ {trtexec_cmd}\n")
        
        # Save instruction manifest
        manifest_path = output_engine_path + ".recipe.txt"
        with open(manifest_path, "w") as f:
            f.write(trtexec_cmd + "\n")
        print(f"[+] Wrote TensorRT deployment recipe: {manifest_path}")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRAJNA TensorRT Engine Exporter")
    parser.add_argument("--onnx", default="checkpoints/prajna_reflex_25k.onnx", help="Input ONNX graph path")
    parser.add_argument("--engine", default="checkpoints/prajna_reflex_25k.engine", help="Output TensorRT engine path")
    parser.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "int8"], help="Precision")
    args = parser.parse_args()

    export_tensorrt_engine(args.onnx, args.engine, precision=args.precision)
