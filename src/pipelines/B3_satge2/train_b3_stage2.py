import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from torch.utils.tensorboard import SummaryWriter

from src.datasets.volleyball_feature_dataset import VolleyballB3_stage2
from src.models.b3_b import Stage2Classifier
from src.utils.label_encoder import LabelEncoder
from src.utils.set_seed import set_seed
from src.utils.logger import setup_logger
from src.engine.trainer import train 
from src.mlflow.logger import start_mlflow, end_mlflow


def train_stage2(cfg):

    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Directories
    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["checkpoints_dir"], exist_ok=True)

    logger = setup_logger(
        os.path.join(cfg["output"]["results_dir"], "logs"),
        "train_stage2"
    )

    writer = SummaryWriter(
        log_dir=os.path.join(cfg["output"]["results_dir"], "tensorboard")
    )

    logger.info("Starting B3 Training")
    logger.info(f"Device: {device}")

    # Dataset & Dataloader
    encoder = LabelEncoder(cfg["labels"]["class_names"])

    feature_root = cfg["feature_path"]
    pickle_file = cfg["data"]["annot_file"]

    train_dataset = VolleyballB3_stage2(
        pickle_file=pickle_file,
        videos_root=os.path.join(cfg["data"]["videos_dir"], "videos"),
        video_list=[str(v) for v in cfg["data"]["splits"]["train"]],
        feuture_root=feature_root,
        encoder=encoder
    )

    val_dataset = VolleyballB3_stage2(
        pickle_file=pickle_file,
        videos_root=os.path.join(cfg["data"]["videos_dir"], "videos"),
        video_list=[str(v) for v in cfg["data"]["splits"]["val"]],
        feuture_root=feature_root,
        encoder=encoder
    )

    # Class Weights
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

    # Weighted sampler
    sample_weights = [class_weights[label].item() for label in labels_all]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        sampler=sampler,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True
    )

    logger.info(f"Train class distribution: {counter}")
    logger.info(f"Class weights: {class_weights}")

    # Model / Loss / Optimizer
    model = Stage2Classifier(
        input_dim=2048,
        hidden_dim=1024,
        num_classes=num_classes,
        dropout=0.5
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    # Scheduler
    if cfg["training"]["scheduler"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.1
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.3, patience=2, min_lr=1e-6, threshold=1e-3
        )

    # Start MLflow
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
        mixup=False 
    )

    end_mlflow()
    writer.close()
    logger.info("Stage 2 Training Finished")
