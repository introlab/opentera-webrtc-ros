import argparse
import os

import torch
import torch.nn as nn

from datasets.classification_image_net import CLASS_COUNT
from models.yunet_classifier import YuNetClassifier
from trainers import BackboneTrainer, BackboneDistillationTrainer
from program_arguments import save_arguments, print_arguments


def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)

    parser.add_argument('--channel_scale', type=int, help='Choose the channel scale', required=True)
    parser.add_argument('--activation', choices=['relu', 'silu'], help='Choose the activation', required=True)
    parser.add_argument('--image_height', type=int, help='Choose the image height', required=True)
    parser.add_argument('--image_width', type=int, help='Choose the image width', required=True)

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

    model = create_model(args.channel_scale, args.activation)
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    image_size = (args.image_height, args.image_width)

    output_path = os.path.join(args.output_path, 's' + str(args.channel_scale) + '_' + args.activation +
                               '_' + str(args.image_width) + 'x' + str(args.image_height) +
                               '_' + str(args.learning_rate) + '_wd' + str(args.weight_decay) +
                               '_ts' + str(args.teacher_channel_scale) + '_a' + str(args.distillation_loss_alpha))
    save_arguments(output_path, args)
    print_arguments(args)

    if args.teacher_channel_scale is not None and args.teacher_model_checkpoint is not None:
        teacher_model = create_model(args.teacher_channel_scale, args.activation)
        trainer = BackboneDistillationTrainer(device, model, teacher_model,
                                              dataset_root=args.dataset_root,
                                              output_path=output_path,
                                              epoch_count=args.epoch_count,
                                              learning_rate=args.learning_rate,
                                              weight_decay=args.weight_decay,
                                              batch_size=args.batch_size,
                                              image_size=image_size,
                                              student_model_checkpoint=args.model_checkpoint,
                                              teacher_model_checkpoint=args.teacher_model_checkpoint,
                                              loss_alpha=args.distillation_loss_alpha)
        trainer.train()
    elif args.teacher_channel_scale is not None or args.teacher_model_checkpoint is not None:
        raise ValueError('teacher_channel_scale and teacher_model_checkpoint must be set.')
    else:
        trainer = BackboneTrainer(device, model,
                                  dataset_root=args.dataset_root,
                                  output_path=output_path,
                                  epoch_count=args.epoch_count,
                                  learning_rate=args.learning_rate,
                                  weight_decay=args.weight_decay,
                                  batch_size=args.batch_size,
                                  image_size=image_size,
                                  model_checkpoint=args.model_checkpoint
                                  )
        trainer.train()


def create_model(channel_scale, activation_name):
    activation = activation_name_to_class(activation_name)
    return YuNetClassifier(class_count=CLASS_COUNT, channel_scale=channel_scale, activation=activation)


def activation_name_to_class(name):
    if name == 'relu':
        return nn.ReLU
    elif name == 'silu':
        return nn.SiLU
    else:
        raise ValueError('Invalid activation name')


if __name__ == '__main__':
    main()
