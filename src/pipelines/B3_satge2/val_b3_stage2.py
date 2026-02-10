import os
import torch
from torch.utils.data import DataLoader

from src.datasets.volleyball_feature_dataset import VolleyballB3_stage2
from src.models.b3_b import Stage2Classifier
from src.utils.label_encoder import LabelEncoder
from src.utils.logger import setup_logger
from src.utils.set_seed import set_seed
from src.engine.evaluator import evaluate
from src.mlflow.logger import start_mlflow, end_mlflow


def eval_b3_stage2(cfg):
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Logger
    logger = setup_logger(
        os.path.join(cfg["output"]["results_dir"], "logs"),
        "eval_b3_stage2"
    )
    logger.info("Starting B3 Stage2 TEST evaluation")
    logger.info(f"Device: {device}")

    # Label Encoder
    encoder = LabelEncoder(cfg["labels"]["class_names"])

    # Dataset & Loader
    feature_root = cfg["feature_path"]
    pickle_file = cfg["data"]["annot_file"]

    test_dataset = VolleyballB3_stage2(
        pickle_file=pickle_file,
        videos_root=os.path.join(cfg["data"]["videos_dir"], "videos"),
        video_list=[str(v) for v in cfg["data"]["splits"]["test"]],
        feuture_root=feature_root,
        encoder=encoder
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True
    )

    # Model
    num_classes = len(encoder.classes_)
    model = Stage2Classifier(
        input_dim=4096,
        hidden_dim=2048,
        num_classes=num_classes,
        dropout=0.5
    ).to(device)

    # Load checkpoint
    checkpoint = os.path.join(cfg["output"]["checkpoints_dir"], "best.pt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # Start MLflow
    start_mlflow(cfg["baseline"] + "_test", cfg["output"]["mlruns_dir"])

    # Evaluate
    evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        cfg=cfg,
        logger=logger,
        encoder=encoder,
        split_name="test"
    )

    # End MLflow
    end_mlflow()
    logger.info("B3 Stage2 TEST evaluation finished successfully")
