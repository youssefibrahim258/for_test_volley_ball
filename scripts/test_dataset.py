
# python -m scripts.test_dataset
import matplotlib.pyplot as plt
import yaml
import os
from torchvision import transforms
from src.datasets.volleyball_player_dataset import VolleyballB3Dataset
from src.utils.label_encoder import LabelEncoder

with open("configs/B3.yaml") as f:
        cfg = yaml.safe_load(f)

def visualize_classes(dataset, encoder, max_images_per_class=5):
    """
    عرض أمثلة من كل class.
    - dataset: الـ dataset object
    - encoder: الـ label encoder
    - max_images_per_class: عدد الصور اللي عايز تظهر لكل class
    """
    # جهز dict لتخزين الصور لكل class
    class_images = {cls: [] for cls in encoder.classes_}

    # loop على dataset
    for i in range(len(dataset)):
        img, label_id = dataset[i]
        label_name = encoder.decode(label_id)

        if len(class_images[label_name]) < max_images_per_class:
            class_images[label_name].append(img)

        # اكسر لو جمعنا العدد المطلوب لكل class
        if all(len(v) >= max_images_per_class for v in class_images.values()):
            break

    # عرض الصور
    for cls, imgs in class_images.items():
        plt.figure(figsize=(15, 3))
        plt.suptitle(cls, fontsize=16)
        for idx, img in enumerate(imgs):
            img_np = img.permute(1, 2, 0).numpy()
            plt.subplot(1, max_images_per_class, idx + 1)
            plt.imshow(img_np)
            plt.axis('off')
        plt.show()


def main():
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    class_names = ["waiting", "setting", "digging", "falling", "spiking", "blocking",
                   "jumping", "moving", "standing"]
    encoder = LabelEncoder(class_names=class_names)

    dataset = VolleyballB3Dataset(
        pickle_file=cfg["data"]["annot_file"],
        videos_root=os.path.join(cfg["data"]["videos_dir"],"videos"),
        video_list=["7","10"],
        encoder=encoder,
        transform=transform
    )

    print("Dataset size:", len(dataset))

    visualize_classes(dataset, encoder, max_images_per_class=5)


if __name__ == "__main__":

    main()


