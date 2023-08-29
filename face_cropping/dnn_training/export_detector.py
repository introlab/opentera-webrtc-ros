import argparse

import torch

from train_detector import create_model

from modules import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Export Detector')
    parser.add_argument('--channel_scale', type=float, help='Choose the channel scale', required=True)
    parser.add_argument('--head_kernel_size', type=int, help='Choose the head kernel size', required=True)
    parser.add_argument('--activation', choices=['relu', 'leaky_relu', 'silu'], help='Choose the activation',
                        required=True)
    parser.add_argument('--image_size', type=int, help='Choose the image width and height', required=True)

    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output file path', required=True)

    args = parser.parse_args()

    model = create_model(args.channel_scale, args.head_kernel_size, args.activation, output_decoded_predictions=True)
    load_checkpoint(model, args.model_checkpoint)
    model.eval()

    x = torch.rand(1, 3, args.image_size, args.image_size)
    jit_model = torch.jit.trace(model, x)
    jit_model.save(args.output_path)

    if torch.cuda.is_available():
        jit_model = torch.jit.trace(model.cuda(), x.cuda())
        jit_model.save(args.output_path + '.cuda')


if __name__ == '__main__':
    main()
