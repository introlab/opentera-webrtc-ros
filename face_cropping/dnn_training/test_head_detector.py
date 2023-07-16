import argparse

import torch

import cv2

from modules import load_checkpoint
from train_head_detector import create_model, BACKBONE_TYPES


def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--backbone_type', choices=BACKBONE_TYPES, help='Choose the backbone type', required=True)
    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', required=True)
    parser.add_argument('--video_device_id', type=int, help='Choose the video device id', required=True)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    model = create_model(args.backbone_type)
    load_checkpoint(model, args.model_checkpoint)
    model = model.to(device)
    model.eval()

    test(model, device, args.video_device_id)


def test(model, device, video_device_id):
    video_capture = cv2.VideoCapture(video_device_id)

    while True:
        ok, frame = video_capture.read()
        if not ok:
            continue

        draw_head_box(frame, model, device)

        cv2.imshow('video_{}'.format(video_device_id), frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    video_capture.release()
    cv2.destroyAllWindows()


def draw_head_box(frame, model, device):
    pass  # TODO


if __name__ == '__main__':
    main()
