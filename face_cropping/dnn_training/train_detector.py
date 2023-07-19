import argparse
import os

import torch

from datasets.classification_image_net import CLASS_COUNT
from models import Detector
from modules.backbones import YuNetBackbone
from modules.necks import YunetFpn
from modules.heads import YunetHead
from trainers import DetectorTrainer

from train_backbone import activation_name_to_class
from program_arguments import save_arguments, print_arguments


def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--dataset_type', choices=['wider_face', 'open_images_head'],
                        help='Choose the dataset type', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)

    parser.add_argument('--channel_scale', type=int, help='Choose the channel scale', required=True)
    parser.add_argument('--activation', choices=['relu', 'silu'], help='Choose the activation', required=True)
    parser.add_argument('--image_size', type=int, help='Choose the image width and height', required=True)

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--weight_decay', type=float, help='Choose the weight decay', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)

    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)

    args = parser.parse_args()

    model = create_model(args.channel_scale, args.activation)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    image_size = (args.image_size, args.image_size)

    output_path = os.path.join(args.output_path, 's' + str(args.channel_scale) + '_' + args.activation +
                               '_' + str(image_size[0]) + 'x' + str(image_size[1]) +
                               '_' + str(args.learning_rate) + '_wd' + str(args.weight_decay))
    save_arguments(output_path, args)
    print_arguments(args)

    trainer = DetectorTrainer(device, model,
                              dataset_root=args.dataset_root,
                              dataset_type=args.dataset_type,
                              output_path=output_path,
                              epoch_count=args.epoch_count,
                              learning_rate=args.learning_rate,
                              weight_decay=args.weight_decay,
                              batch_size=args.batch_size,
                              image_size=image_size,
                              model_checkpoint=args.model_checkpoint)
    trainer.train()


def create_model(channel_scale, activation_name, output_decoded_predictions=False):
    activation = activation_name_to_class(activation_name)
    backbone = YuNetBackbone(activation=activation, channel_scale=channel_scale)
    neck = YunetFpn(backbone.output_channels(), activation=activation)
    head = YunetHead(backbone.output_channels(), backbone.output_strides(), activation)
    return Detector(backbone=backbone, neck=neck, head=head, output_decoded_predictions=output_decoded_predictions)


if __name__ == '__main__':
    main()
