"""Training entrypoint (Hydra-powered)."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ..utils import configure_logging, get_logger, seed_everything

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logging(level=cfg.get("log_level", "INFO"))
    seed_everything(cfg.get("seed", 42))
    log.info("train.start", config=OmegaConf.to_container(cfg, resolve=True))

    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import MLFlowLogger

    from ..data import SegmentationDataModule
    from ..models import SegmentationModule, build_model

    dm = SegmentationDataModule(**cfg.data)
    net = build_model(cfg.model.name, num_classes=cfg.model.num_classes)
    lit = SegmentationModule(
        net,
        num_classes=cfg.model.num_classes,
        lr=cfg.model.lr,
        model_name=cfg.model.name,
    )

    out_dir = Path(cfg.trainer.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=out_dir / "checkpoints",
            filename="best",
            monitor=cfg.trainer.monitor,
            mode=cfg.trainer.monitor_mode,
            save_top_k=1,
        ),
        EarlyStopping(
            monitor=cfg.trainer.monitor,
            mode=cfg.trainer.monitor_mode,
            patience=cfg.trainer.patience,
        ),
    ]
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.experiment_name, tracking_uri=cfg.trainer.tracking_uri
    )
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=callbacks,
        logger=mlflow_logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        deterministic="warn",
    )
    trainer.fit(lit, dm)
    log.info("train.done", ckpt=str(out_dir / "checkpoints" / "best.ckpt"))


if __name__ == "__main__":
    main()
