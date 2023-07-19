import argparse

import torch

import cv2
from PIL import Image

from datasets.detection_transforms import DetectionValidationTransform
from modules import load_checkpoint
from modules.heads import filter_decoded_bboxes, TL_X_INDEX, TL_Y_INDEX, BR_X_INDEX, BR_Y_INDEX
from train_detector import create_model


def main():
    parser = argparse.ArgumentParser(description='Train Backbone')
    parser.add_argument('--use_gpu', action='store_true', help='Use the GPU')
    parser.add_argument('--channel_scale', type=int, help='Choose the channel scale', required=True)
    parser.add_argument('--activation', choices=['relu', 'silu'], help='Choose the activation', required=True)
    parser.add_argument('--image_size', type=int, help='Choose the image width and height', required=True)
    parser.add_argument('--model_checkpoint', type=str, help='Choose the model checkpoint file', required=True)
    parser.add_argument('--video_device_id', type=int, help='Choose the video device id', required=True)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    model = create_model(args.channel_scale, args.activation, output_decoded_predictions=True)
    load_checkpoint(model, args.model_checkpoint)
    model = model.to(device)
    model.eval()

    transform = DetectionValidationTransform((args.image_size, args.image_size))

    test(model, device, transform, args.video_device_id)


def test(model, device, transform, video_device_id):
    video_capture = cv2.VideoCapture(video_device_id)

    while True:
        ok, frame = video_capture.read()
        if not ok:
            continue

        draw_head_box(frame, model, device, transform)

        cv2.imshow('video_{}'.format(video_device_id), frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    video_capture.release()
    cv2.destroyAllWindows()


def draw_head_box(cv_image, model, device, transform):
    with torch.no_grad():
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        tensor, scale = transform(pil_image)
        bboxes = model(tensor.unsqueeze(0).to(device))[0]
        filtered_bboxes = filter_decoded_bboxes(bboxes, confidence_threshold=0.9)

        for bbox in filtered_bboxes:
            tl_x = int(bbox[TL_X_INDEX].item() / scale)
            tl_y = int(bbox[TL_Y_INDEX].item() / scale)
            br_x = int(bbox[BR_X_INDEX].item() / scale)
            br_y = int(bbox[BR_Y_INDEX].item() / scale)
            cv2.rectangle(cv_image, (tl_x, tl_y), (br_x, br_y), (255, 0, 0), 1)


if __name__ == '__main__':
    main()
