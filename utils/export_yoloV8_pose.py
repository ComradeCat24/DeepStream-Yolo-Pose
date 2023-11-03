import os
import sys
import argparse
import warnings
import onnx
import torch
import torch.nn as nn
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.modules import C2f, Detect, RTDETRDecoder


class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.transpose(1, 2)
        boxes = x[:, :, :4]
        scores = x[:, :, 4:5]
        kpts = x[:, :, 5:]
        return boxes, scores, kpts


def suppress_warnings():
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)


def fuse_conv_and_bn(conv, bn):
    # Extract Conv and BatchNorm weights and bias
    w_conv, b_conv = conv.weight, conv.bias
    w_bn, b_bn = bn.weight, bn.bias
    mean, var = bn.running_mean, bn.running_var

    # Calculate the new weights and bias
    w_fused = w_conv * (w_bn / torch.sqrt(var + bn.eps))
    b_fused = (b_conv - mean) * (w_bn / torch.sqrt(var + bn.eps)) + b_bn

    # Update Conv layer weights and bias
    conv.weight = torch.nn.Parameter(w_fused, requires_grad=False)
    conv.bias = torch.nn.Parameter(b_fused, requires_grad=False)

    return conv


def yolov8_export(weights, device):
    model = YOLO(weights)
    model = deepcopy(model.model).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    if args.half:
        model = model.half()
        # Fuse Conv and BatchNorm layers
        for k, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and isinstance(m, nn.BatchNorm2d):
                m = fuse_conv_and_bn(m, m)
    else:
        model.float()
    for k, m in model.named_modules():
        if isinstance(m, (Detect, RTDETRDecoder)):
            m.dynamic = False
            m.export = True
            m.format = 'onnx'
        elif isinstance(m, C2f):
            m.forward = m.forward_split
    return model

def main(args):
    suppress_warnings()

    print(f'Starting {args.weights} model conversions...')

    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    print(f'Selected device: {device}')  

    model = yolov8_export(args.weights, device)

    model = nn.Sequential(model, DeepStreamOutput())

    img_size = args.size * 2 if len(args.size) == 1 else args.size

    target_dtype = torch.float16 if args.half else torch.float32
    onnx_input_img = torch.zeros(args.batch, 3, *img_size, dtype=target_dtype).to(device)

    onnx_output_file = os.path.basename(args.weights).split('.pt')[0] + '.onnx'

    dynamic_axes = {
        'input': {0: 'batch'},
        'boxes': {0: 'batch'},
        'scores': {0: 'batch'},
        'kpts': {0: 'batch'}
    }

    print(f'Exporting the model to ONNX in {"half-precision (float16)" if args.half else "full-precision (float32)"}...')
    torch.onnx.export(
        model, 
        onnx_input_img, 
        onnx_output_file, 
        verbose=False, 
        opset_version=args.opset,
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['boxes', 'scores', 'kpts'],
        dynamic_axes=dynamic_axes if args.dynamic else None
    )

    if args.simplify:
        print('Simplifying the ONNX model...')
        import onnxsim
        model_onnx = onnx.load(onnx_output_file)
        model_onnx, _ = onnxsim.simplify(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print(f'Model conversion complete. ONNX model saved as {onnx_output_file}')


def parse_args():
    parser = argparse.ArgumentParser(description='DeepStream YOLOv8-Pose conversion')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--opset', type=int, default=16, help='ONNX opset version')
    parser.add_argument('--half', action='store_true', help='Use half-precision (float16)')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    source_group.add_argument('--batch', type=int, default=1, help='Static batch-size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0, 1, 2, 3 or cpu')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
