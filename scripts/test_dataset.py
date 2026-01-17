import matplotlib.pyplot as plt
from torchvision import transforms

from src.datasets.volleyball_player_dataset import VolleyballB3Dataset
from src.utils.label_encoder import LabelEncoder
from torchvision import transforms


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomRotation(degrees=3),
        # transforms.RandomAffine(degrees=0, translate=(0.03,0.03), scale=(0.97,1.03)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    class_names = ["waiting", "setting", "digging", "falling", "spiking", "blocking", "jumping", "moving", "standing"]


    encoder = LabelEncoder(class_names=class_names)


    dataset = VolleyballB3Dataset(
        pickle_file=r"data_set/annot_all.pkl",
        videos_root=r"data_set\videos",
        video_list=["7"],
        encoder=encoder,
        transform=transform
    )

    print("Dataset size:", len(dataset))

    img_crop, label_id1 = dataset[8]
    print(label_id1)
    label_name = encoder.decode(label_id1)
    img = img_crop.permute(1, 2, 0).numpy()  

    plt.imshow(img)
    plt.title(label_name)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
