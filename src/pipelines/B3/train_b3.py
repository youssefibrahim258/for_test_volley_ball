import os
import torch
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn

from src.utils.set_seed import set_seed
from src.utils.logger import setup_logger
from src.datasets.volleyball_player_dataset import VolleyballB3Dataset
from src.utils.label_encoder import LabelEncoder
from src.models.b1_resnet import ResNetB1
from src.mlflow.logger import start_mlflow, end_mlflow
from src.engine.trainer import train
from src.utils.focal_loss import FocalLoss



def train_b3(cfg):
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["checkpoints_dir"], exist_ok=True)

    logger = setup_logger(os.path.join(cfg["output"]["results_dir"], "logs"), "train_b3")
    writer = SummaryWriter(log_dir=os.path.join(cfg["output"]["results_dir"], "tensorboard"))

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomRotation(degrees=3),
        transforms.RandomAffine(degrees=0, translate=(0.03,0.03), scale=(0.97,1.03)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Dataset & Encoder
    encoder = LabelEncoder(cfg["labels"]["class_names"])
    pickle_file = cfg["data"]["annot_file"]
    videos_root = os.path.join(cfg["data"]["videos_dir"], "videos")
    train_videos_list = [str(v) for v in cfg["data"]["splits"]["train"]]
    val_videos_list = [str(v) for v in cfg["data"]["splits"]["val"]]

    train_dataset = VolleyballB3Dataset(
        pickle_file,
        videos_root,
        train_videos_list,
        encoder,
        train_transform
    )
    # print(Counter(train_dataset.labels))


    val_dataset = VolleyballB3Dataset(
        pickle_file,
        videos_root,
        val_videos_list,
        encoder,
        val_transform
    )

    # Class weights
    labels_all = [label for _, label in train_dataset]
    counter = Counter(labels_all)

    num_classes = len(encoder.classes_)
    total_samples = sum(counter.values())

    class_weights = [
        total_samples / (num_classes * counter[i])
        if counter[i] > 0 else 0.0
        for i in range(num_classes)
    ]

    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # WeightedRandomSampler
    sample_weights = [class_weights[label].item() for label in labels_all]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        sampler=sampler,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        persistent_workers=True
    )
    

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        persistent_workers=True
    )

    # Model, criterion, optimizer
    model = ResNetB1(num_classes=cfg["num_classes"]).to(device)
    criterion = FocalLoss(gamma=2.0, weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1
    )

    # Logging
    logger.info("Starting B3 Training")
    logger.info(f"Device: {device}")
    logger.info(f"Train class distribution: {counter}")
    logger.info(f"Class weights: {class_weights}")

    start_mlflow(cfg["baseline"], cfg["output"]["mlruns_dir"])
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        cfg=cfg,
        logger=logger,
        writer=writer,
        encoder=encoder,
        mixup=cfg["training"]["mixup"]
    )
    end_mlflow()

    writer.close()
    logger.info("B3 Training Finished")
