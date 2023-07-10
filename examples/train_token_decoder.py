
import hydra
from omegaconf import DictConfig
from lightning.pytorch import Trainer, seed_everything
import torch

from mqtts.data.datamodule import MQTTSDataModule
from mqtts.model.token_decoder_lightning_module import TokenDecoderLightningModule


@hydra.main(version_base="1.3", config_name="config", config_path="./configs")
def main(cfg: DictConfig):
    seed_everything(1234)
    torch.set_float32_matmul_precision('medium') 
    datamodule = MQTTSDataModule(cfg)
    lightning_module = TokenDecoderLightningModule(cfg,len(datamodule.vocab))
    loggers = hydra.utils.instantiate(cfg.train.token_decoder.loggers)
    trainer = hydra.utils.instantiate(cfg.train.token_decoder.trainer,logger=loggers)
    trainer.fit(lightning_module,datamodule)


if __name__ == "__main__":
    main()
