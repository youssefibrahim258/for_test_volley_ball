# python -m src.utils_data.features_extraction

import os
import numpy as np
import torch
import yaml
from PIL import Image

import torch.nn as nn
import torchvision.transforms as transforms
from src.utils_data.annotations import load_tracking_annot
from torchvision import models


class ResNetB3(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        return self.model(x)



with open("configs/B3_b.yaml") as f:
    cfg = yaml.safe_load(f)

dataset_root = cfg["data"]["videos_dir"]

def check():
    print('torch: version', torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available.")
        num_devices = torch.cuda.device_count()
        print(f"Number of GPU devices: {num_devices}")
        for i in range(num_devices):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print("Current device:", torch.cuda.current_device())
    else:
        print("CUDA is not available. Using CPU.")



def prepare_model(image_level=False):
    if image_level:
        # image has a lot of space around objects. Let's crop around
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        # already croped box. Don't crop more
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNetB3(num_classes=9)

    state_dict = torch.load(cfg["model_path"], map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.model.fc = nn.Identity()

    model.to(device)
    model.eval()

    return model, preprocess, device



def extract_features(clip_dir_path, annot_file, output_file, model, preprocess, device, image_level=False):
    frame_boxes = load_tracking_annot(annot_file)

    if len(frame_boxes) == 0:
        return

    frame_ids = sorted(frame_boxes.keys())
    mid_idx = len(frame_ids) // 2
    mid_frame_id = frame_ids[mid_idx]

    boxes_info = frame_boxes[mid_frame_id]

    all_features = []
    all_frame_ids = []
    all_player_ids = []

    with torch.no_grad():
        try:
            img_path = os.path.join(clip_dir_path, f'{mid_frame_id}.jpg')
            image = Image.open(img_path).convert('RGB')

            if image_level:
                x = preprocess(image).unsqueeze(0).to(device)
                feats = model(x)
                feats = feats.view(1, -1)

                all_features.append(feats.cpu().numpy())
                all_frame_ids.append(mid_frame_id)
                all_player_ids.append(0)  # whole image

            else:
                crops = []
                for idx, box_info in enumerate(boxes_info):
                    x1, y1, x2, y2 = box_info.box
                    crop = image.crop((x1, y1, x2, y2))
                    crops.append(preprocess(crop).unsqueeze(0))

                if len(crops) == 0:
                    return

                x = torch.cat(crops).to(device)
                feats = model(x)
                feats = feats.squeeze(-1).squeeze(-1)  # (num_players, 2048)

                all_features.append(feats.cpu().numpy())
                all_frame_ids.extend([mid_frame_id] * feats.size(0))
                all_player_ids.extend(list(range(len(crops))))

        except Exception as e:
            print(f"Error in frame {mid_frame_id}: {e}")
            return

    if len(all_features) > 0:
        all_features = np.concatenate(all_features, axis=0)
        all_frame_ids = np.array(all_frame_ids)
        all_player_ids = np.array(all_player_ids)

        np.savez(
            output_file,
            features=all_features,
            frame_ids=all_frame_ids,
            player_ids=all_player_ids
        )


if __name__ == '__main__':
    check()

    image_level = False
    model, preprocess, device = prepare_model(image_level)

    videos_root = f'{dataset_root}/videos'
    annot_root = f'{dataset_root}/volleyball_tracking_annotation'
    output_root = f'{dataset_root}/features/image-level/resnet'

    videos_dirs = sorted(os.listdir(videos_root))

    for idx, video_dir in enumerate(videos_dirs):
        video_dir_path = os.path.join(videos_root, video_dir)
        if not os.path.isdir(video_dir_path):
            continue

        print(f'{idx}/{len(videos_dirs)} - Processing Dir {video_dir_path}')

        clips_dir = sorted(os.listdir(video_dir_path))

        for clip_dir in clips_dir:
            clip_dir_path = os.path.join(video_dir_path, clip_dir)
            if not os.path.isdir(clip_dir_path):
                continue

            print(f'\t{clip_dir_path}')

            annot_file = os.path.join(
                annot_root, video_dir, clip_dir, f'{clip_dir}.txt'
            )

            out_dir = os.path.join(output_root, video_dir)
            os.makedirs(out_dir, exist_ok=True)

            output_file = os.path.join(out_dir, f'{clip_dir}.npz')

            extract_features(
                clip_dir_path,
                annot_file,
                output_file,
                model,
                preprocess,
                device,
                image_level=image_level
            )
