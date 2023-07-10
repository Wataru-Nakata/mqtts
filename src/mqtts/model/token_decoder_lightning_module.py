
from lightning.pytorch import LightningModule
import torch
from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss, Embedding
import hydra
from lightning.pytorch import loggers
from .modules.wildttstransformer import TTSDecoder
from matplotlib import pyplot as plt

from mqtts.model.quantizer_lightning_module import QuantizerLightningModule

class TokenDecoderLightningModule(LightningModule):
    def __init__(self, cfg:DictConfig,phoneset_size) -> None:
        super().__init__()
        self.token_decoder = TTSDecoder(cfg.model.token_decoder.decoder,phoneset_size)
        self.n_codes = self.token_decoder.transducer.n_decoder_codes

        self.cross_entropy = CrossEntropyLoss(label_smoothing=cfg.model.token_decoder.label_smoothing)
        self.phone_embedding = Embedding(phoneset_size,cfg.model.token_decoder.decoder.hidden_size,padding_idx=0)
        self.spkr_embedding = Embedding(101,cfg.model.token_decoder.decoder.hidden_size)
        quantizer = QuantizerLightningModule.load_from_checkpoint(cfg.data.quantizer_path,cfg=cfg)
        self.vocoder = quantizer.generator
        self.vocoder_spkr_embedding = quantizer.speaker_embedding
        self.vocoder_embedder = quantizer.quantizer
        self.cfg = cfg
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg.model.token_decoder.optim,params=self.parameters())
        #scheduler = hydra.utils.instantiate(self.cfg.model.token_decoder.scheduler,params=self.parameters())
        return optimizer
    def training_step(self, batch,batch_idx):
        speaker_embedding = self.spkr_embedding(batch['speaker'])
        phone_features = self.phone_embedding(batch['phones'])
        recons_segments = self.token_decoder(batch['input_tokens'], phone_features,speaker_embedding,batch['tokens_mask'], batch['phones_mask'])
        targets = recons_segments['logits'][~batch['tokens_mask']].view(-1,self.n_codes)
        labels = batch['output_tokens'][~batch['tokens_mask']].view(-1)
        loss = self.cross_entropy(targets,labels)
        acc = (targets.argmax(-1) == labels).float().mean()
        self.log("train/loss", loss)
        self.log('train/acc', acc)
        return loss
    def validation_step(self,batch,batch_idx):
        speaker_embedding = self.spkr_embedding(batch['speaker'])
        phone_features = self.phone_embedding(batch['phones'])
        recons_segments = self.token_decoder(batch['input_tokens'], phone_features,speaker_embedding,batch['tokens_mask'], batch['phones_mask'])
        targets = recons_segments['logits'][~batch['tokens_mask']].view(-1,self.n_codes)
        labels = batch['output_tokens'][~batch['tokens_mask']].view(-1)
        loss = self.cross_entropy(targets,labels)
        acc = (targets.argmax(-1) == labels).float().mean()
        self.log("val/loss", loss)
        self.log('val/acc', acc)

        if batch_idx < self.cfg.model.token_decoder.logging_samples:
            phone_mask = torch.full((phone_features.size(0),phone_features.size(1)),False,dtype=torch.bool,device=phone_features.device)
            synthetic, infer_attn = self.token_decoder.inference_topkp_sampling_batch(phone_features,speaker_embedding,phone_mask,prior=None,output_alignment=True)
            synthetic = synthetic[0].unsqueeze(0)
            synthetic = self.infer_vocoder(synthetic,batch['speaker'])
            gt_speech = self.infer_vocoder(batch['input_tokens'][:,1:],batch['speaker'])
            self.log_audio(synthetic.cpu().view(-1),f"synthetic/{batch['filenames'][0]}",self.cfg.sample_rate)
            self.log_audio(gt_speech.cpu().view(-1),f"gt/{batch['filenames'][0]}",self.cfg.sample_rate)
            fig= plt.figure(figsize=(10,10))
            plt.imshow(infer_attn.cpu().float()[0].T,origin='lower')
            plt.legend()
            self.log_plot(fig,"val/attention")
        return loss
    def infer_vocoder(self,tokens,speaker):
        vocoder_speaker_embedding = self.vocoder_spkr_embedding(speaker)
        feature = self.vocoder_embedder.embed(tokens)
        speech = self.vocoder(feature,vocoder_speaker_embedding)
        return speech


    def log_audio(self, audio, name, sampling_rate):
        audio = audio.float()
        for logger in self.loggers:
            match type(logger):
                case loggers.WandbLogger:
                    import wandb

                    wandb.log(
                        {name: wandb.Audio(audio, sample_rate=sampling_rate)},
                        step=self.global_step,
                    )
                case loggers.TensorBoardLogger:
                    logger.experiment.add_audio(
                        name,
                        audio,
                        self.global_step,
                        sampling_rate,
                    )
    def log_plot(self,fig,name):
        for logger in self.loggers:
            match type(logger):
                case loggers.WandbLogger:
                    import wandb

                    wandb.log(
                        {name: wandb.Image(fig)},
                        step=self.global_step,
                    )

