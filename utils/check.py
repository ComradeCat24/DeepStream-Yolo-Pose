import onnx
import os
import argparse

# Define a dictionary to map ONNX data types to human-readable names
data_type_mapping = {
    1: "Float (FP32)",
    10: "Float16 (FP16)",
    11: "Int8",
    12: "Int16",
    13: "Int32",
    14: "Int64",
    15: "String",
    16: "Bool",
    17: "Float (BF16)",
    18: "Double (FP64)",
}

# Function to get human-readable data type
def get_human_readable_data_type(tensor):
    return data_type_mapping.get(tensor.type.tensor_type.elem_type, "Unknown")

def parse_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Check the data types of input and output tensors in an ONNX model")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the ONNX model file")
    args = parser.parse_args()
    if not os.path.isfile(args.model_path):
        raise SystemExit('Invalid model path file')
    return args

def main():
    try:
        # Load the ONNX model from the specified file
        onnx_model = onnx.load(args.model_path)

        # Check the data types of input tensors
        for input_info in onnx_model.graph.input:
            data_type = get_human_readable_data_type(input_info)
            print(f"Input tensor '{input_info.name}' data type: {data_type}")

        # Check the data types of output tensors
        for output_info in onnx_model.graph.output:
            data_type = get_human_readable_data_type(output_info)
            print(f"Output tensor '{output_info.name}' data type: {data_type}")

    except FileNotFoundError:
        print("Error: Model file not found.")
    except onnx.onnx_cpp2py_export.InvalidProtoError:
        print("Error: The specified file is not a valid ONNX model.")

if __name__ == '__main__':
    args = parse_args()
    main()
