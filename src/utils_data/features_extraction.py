from torchvision.transforms import transforms
from torchvision.models import resnet50
import torch
import torch.nn as nn



def prepare_model(image_level=False):

    if image_level :
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model=resnet50(pretrained=True)

    model =nn.Sequential(*(list(model.children())[:-1]))

    model.to(device)

    model.eval()

    return model , preprocess

    ...



if __name__ == "__main__":
    image_level=False

    model , preprocess= prepare_model(image_level)