import argparse

import torch
import torch.nn.functional as F

import cv2

from modules import load_checkpoint
from train_head_detector import create_model, BACKBONE_TYPES
from datasets.open_images_head_detector_dataset import NO_HEAD_CLASS_INDEX
from models.head_detector import OUTPUT_CLASS_INDEX_MIN, OUTPUT_CLASS_INDEX_MAX
from models.head_detector import OUTPUT_X_INDEX, OUTPUT_Y_INDEX, OUTPUT_W_INDEX, OUTPUT_H_INDEX
from trainers.head_detector_trainer import IMAGE_SIZE


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
    with torch.no_grad():
        input = torch.from_numpy(frame).to(device).permute(2, 0, 1).float() / 255
        resized_input = F.interpolate(input.unsqueeze(0), size=IMAGE_SIZE)
        output = model(resized_input)[0]

        class_index = output[OUTPUT_CLASS_INDEX_MIN:OUTPUT_CLASS_INDEX_MAX].argmax()
        if class_index == NO_HEAD_CLASS_INDEX:
            return

        cx = output[OUTPUT_X_INDEX].item() * frame.shape[1]
        cy = output[OUTPUT_Y_INDEX].item() * frame.shape[0]
        w = output[OUTPUT_W_INDEX].item() * frame.shape[1]
        h = output[OUTPUT_H_INDEX].item() * frame.shape[0]

    x0 = int(cx - w / 2)
    y0 = int(cy - h / 2)
    x1 = int(cx + w / 2)
    y1 = int(cy + h / 2)

    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)


if __name__ == '__main__':
    main()
