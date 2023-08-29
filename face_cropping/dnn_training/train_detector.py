import argparse
from pathlib import Path

import torch

from models import Detector
from modules.backbones import YuNetBackbone
from modules.necks import YunetFpn
from modules.heads import YunetHead
from trainers import DetectorTrainer, DetectorDistillationTrainer
from trainers.detector_trainer import DatasetType

from train_backbone import ActivationName
from program_arguments import save_arguments, print_arguments


def main():
    parser = argparse.ArgumentParser(description='Train Detector')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--dataset_type', type=DatasetType, choices=list(DatasetType), help='Choose the dataset type',
                        required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)

    parser.add_argument('--channel_scale', type=float, help='Choose the channel scale', required=True)
    parser.add_argument('--head_kernel_size', type=int, help='Choose the head kernel size', required=True)
    parser.add_argument('--activation', type=ActivationName, choices=list(ActivationName), help='Choose the activation',
                        required=True)
    parser.add_argument('--image_size', type=int, help='Choose the image width and height', required=True)

    parser.add_argument('--use_mosaic', action='store_true', help='Use the GPU')

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--weight_decay', type=float, help='Choose the weight decay', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)

    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)

    parser.add_argument('--teacher_channel_scale', type=int, help='Choose the teacher channel scale',
                        default=None)
    parser.add_argument('--teacher_model_checkpoint', type=str, help='Choose the teacher model checkpoint file',
                        default=None)
    parser.add_argument('--distillation_loss_alpha', type=float, help='Choose the alpha for the distillation loss',
                        default=0.25)

    args = parser.parse_args()

    model = create_model(args.channel_scale, args.head_kernel_size, args.activation)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    image_size = (args.image_size, args.image_size)

    output_directory = (f'{args.dataset_type}_s{args.channel_scale}_hk{args.head_kernel_size}_{args.activation}'
                        f'_{image_size[1]}x{image_size[0]}_lr{args.learning_rate}_wd{args.weight_decay}'
                        f'{"_mosaic" if args.use_mosaic else ""}_ts{args.teacher_channel_scale}'
                        f'_a{args.distillation_loss_alpha}')
    output_path = Path(args.output_path) / output_directory
    save_arguments(output_path, args)
    print_arguments(args)

    if args.teacher_channel_scale is not None and args.teacher_model_checkpoint is not None:
        teacher_model = create_model(args.teacher_channel_scale, args.head_kernel_size, args.activation)
        trainer = DetectorDistillationTrainer(device, model, teacher_model,
                                              dataset_root=args.dataset_root,
                                              dataset_type=args.dataset_type,
                                              output_path=output_path,
                                              epoch_count=args.epoch_count,
                                              learning_rate=args.learning_rate,
                                              weight_decay=args.weight_decay,
                                              batch_size=args.batch_size,
                                              image_size=image_size,
                                              use_mosaic=args.use_mosaic,
                                              student_model_checkpoint=args.model_checkpoint,
                                              teacher_model_checkpoint=args.teacher_model_checkpoint,
                                              loss_alpha=args.distillation_loss_alpha)
    elif args.teacher_channel_scale is not None or args.teacher_model_checkpoint is not None:
        raise ValueError('teacher_channel_scale and teacher_model_checkpoint must be set.')
    else:
        trainer = DetectorTrainer(device, model,
                                  dataset_root=args.dataset_root,
                                  dataset_type=args.dataset_type,
                                  output_path=output_path,
                                  epoch_count=args.epoch_count,
                                  learning_rate=args.learning_rate,
                                  weight_decay=args.weight_decay,
                                  batch_size=args.batch_size,
                                  image_size=image_size,
                                  use_mosaic=args.use_mosaic,
                                  model_checkpoint=args.model_checkpoint)
    trainer.train()


def create_model(channel_scale, head_kernel_size, activation_name, output_decoded_predictions=False):
    activation = activation_name.to_class()
    backbone = YuNetBackbone(activation=activation, channel_scale=channel_scale)
    neck = YunetFpn(backbone.output_channels(), activation=activation)
    head = YunetHead(backbone.output_channels(), backbone.output_strides(),
                     head_kernel_size=head_kernel_size, activation=activation)
    return Detector(backbone=backbone, neck=neck, head=head, output_decoded_predictions=output_decoded_predictions)


if __name__ == '__main__':
    main()
