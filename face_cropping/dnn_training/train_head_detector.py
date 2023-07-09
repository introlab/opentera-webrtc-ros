import argparse
import os
import json

import torch

from models import HeadDetector, EfficientNetBackbone
from trainers import HeadDetectorTrainer


BACKBONE_TYPES = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                  'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']


def save_arguments(output_path, args):
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'arguments.json'), 'w') as file:
        json.dump(args.__dict__, file, indent=4, sort_keys=True)


def print_arguments(args):
    print('*******************************************')
    print('**************** Arguments ****************')
    print('*******************************************\n')

    for arg, value in sorted(vars(args).items()):
        print(arg, '=', value)

    print(flush=True)


def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--open_images_root', type=str, help='Choose the Open Images root path', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)
    parser.add_argument('--backbone_type', choices=BACKBONE_TYPES, help='Choose the backbone type', required=True)

    parser.add_argument('--learning_rate', type=float, help='Choose the learning rate', required=True)
    parser.add_argument('--weight_decay', type=float, help='Choose the weight decay', required=True)
    parser.add_argument('--batch_size', type=int, help='Set the batch size for the training', required=True)
    parser.add_argument('--epoch_count', type=int, help='Choose the epoch count', required=True)

    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', default=None)

    args = parser.parse_args()

    model = create_model(args.backbone_type)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    output_path = os.path.join(args.output_path, args.backbone_type + '_lr' + str(args.learning_rate) +
                               '_wd' + str(args.weight_decay))
    save_arguments(output_path, args)
    print_arguments(args)

    trainer = HeadDetectorTrainer(device,
                                  model,
                                  dataset_root=args.open_images_root,
                                  epoch_count=args.epoch_count,
                                  learning_rate=args.learning_rate,
                                  weight_decay=args.weight_decay,
                                  output_path=output_path,
                                  batch_size=args.batch_size,
                                  model_checkpoint=args.model_checkpoint)
    trainer.train()


def create_model(backbone_type):
    if backbone_type.startswith('efficientnet_b'):
        backbone = EfficientNetBackbone(backbone_type, pretrained_backbone=True)
    else:
        raise ValueError('Invalid backbone')

    return HeadDetector(backbone)


if __name__ == '__main__':
    main()
