
import hydra
from omegaconf import DictConfig
from lightning.pytorch import Trainer, seed_everything

import mqtts_lightning
from mqtts_lightning.data.quantizer_datamodule import MQTTSQuantizerDataModule
from mqtts_lightning.model.quantizer_lightning_module import QuantizerLightningModule


@hydra.main(version_base="1.3", config_name="config", config_path="./configs")
def main(cfg: DictConfig):
    seed_everything(1234)
    lightning_module = QuantizerLightningModule(cfg)
    datamodule = MQTTSQuantizerDataModule(cfg)
    loggers = hydra.utils.instantiate(cfg.train.quantizer.loggers)
    trainer = hydra.utils.instantiate(cfg.train.quantizer.trainer,logger=loggers)
    trainer.fit(lightning_module,datamodule)


if __name__ == "__main__":
    main()
