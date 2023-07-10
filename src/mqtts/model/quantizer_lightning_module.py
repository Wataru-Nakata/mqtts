import numpy as np
from lightning.pytorch import LightningModule
from omegaconf import DictConfig
import hydra
from torch import nn as nn
import torch
import itertools
import torchaudio
import lightning.pytorch.loggers as loggers

from mqtts.model.quantizer import Encoder, Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, Quantizer, generator_loss,discriminator_loss,feature_loss


class QuantizerLightningModule(LightningModule):
    def __init__(self, cfg:DictConfig) -> None:
        super().__init__()
        self.encoder = Encoder(cfg.model.quantizer.encoder)
        self.generator = Generator(cfg.model.quantizer.generator)
        self.quantizer = Quantizer(cfg.model.quantizer.quantizer)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        self.automatic_optimization=False
        self.speaker_embedding = nn.Embedding(cfg.model.quantizer.speaker_embedding.n_speakers,cfg.model.quantizer.speaker_embedding.embedding_dim)
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            cfg.sample_rate,
            cfg.model.quantizer.mel.n_fft,
            cfg.model.quantizer.mel.win_length,
            cfg.model.quantizer.mel.hop_length,
            cfg.model.quantizer.mel.f_min,
            cfg.model.quantizer.mel.f_max,
            n_mels=cfg.model.quantizer.mel.n_mels,
        )
        self.reconstruction_loss = nn.MSELoss()
        self.cfg = cfg
        self.save_hyperparameters()
    def forward(self,wav):
        wav = wav.unsqueeze(1)
        c = self.encoder(wav)
        q,loss_q,c = self.quantizer(c)
        return q,loss_q,c

    def training_step(self, batch, batch_idx):
        wav, speaker = batch["resampled_speech.pth"], batch['speaker']
        with torch.no_grad():
            wav_mel = self.calc_logmelspec(wav).detach()

        wav = wav.unsqueeze(1)
        c = self.encoder(wav)

        q, loss_q, c = self.quantizer(c)
        speaker = self.speaker_embedding(speaker)
        wav_generator_out = self.generator(q,speaker)

        opt_g, opt_d = self.optimizers()
        sch_g, sch_d = self.lr_schedulers()
        if self.global_step >= self.cfg.model.quantizer.adversarial_start_step:
            opt_d.zero_grad()

            # mpd
            mpd_out_real, mpd_out_fake, _, _ = self.mpd(
                wav, wav_generator_out.detach()
            )
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                mpd_out_real, mpd_out_fake
            )

            # msd
            msd_out_real, msd_out_fake, _, _ = self.msd(
                wav, wav_generator_out.detach()
            )
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                msd_out_real, msd_out_fake
            )

            loss_disc_all = loss_disc_s + loss_disc_f
            self.manual_backward(loss_disc_all)
            opt_d.step()
            sch_d.step()
            self.log("train/discriminator/loss_disc_f", loss_disc_f)
            self.log("train/discriminator/loss_disc_s", loss_disc_s)
        else:
            loss_disc_f = loss_disc_s = 0.0

        # generator
        opt_g.zero_grad()
        predicted_mel = self.calc_logmelspec(wav_generator_out.squeeze(1))
        loss_recons = self.reconstruction_loss(wav_mel, predicted_mel)
        loss_g = loss_recons * self.cfg.model.quantizer.loss.recons_coef + loss_q * self.cfg.model.quantizer.loss.quantizer_coef
        if self.global_step >self.cfg.model.quantizer.adversarial_start_step:
            (
                mpd_out_real,
                mpd_out_fake,
                fmap_f_real,
                fmap_f_generated,
            ) = self.mpd(wav, wav_generator_out)
            loss_fm_mpd = feature_loss(fmap_f_real, fmap_f_generated)

            # msd
            (
                msd_out_real,
                msd_out_fake,
                fmap_scale_real,
                fmap_scale_generated,
            ) = self.msd(wav, wav_generator_out)
            loss_fm_msd = feature_loss(fmap_scale_real, fmap_scale_generated)

            loss_g_mpd, losses_gen_f = generator_loss(mpd_out_fake)
            loss_g_msd, losses_gen_s = generator_loss(msd_out_fake)
            loss_g += loss_fm_mpd * self.cfg.model.quantizer.loss.fm_mpd_coef
            loss_g += loss_fm_msd * self.cfg.model.quantizer.loss.fm_msd_coef
            loss_g += loss_g_mpd *  self.cfg.model.quantizer.loss.g_mpd_coef
            loss_g += loss_g_msd *  self.cfg.model.quantizer.loss.g_msd_coef
            self.log("train/generator/loss_fm_mpd", loss_fm_mpd)
            self.log("train/generator/loss_fm_msd", loss_fm_msd)
            self.log("train/generator/loss_g_mpd", loss_g_mpd)
            self.log("train/generator/loss_g_msd", loss_g_msd)
        self.manual_backward(loss_g)
        self.log("train/loss_reconstruction", loss_recons)
        self.log("train/loss_quantization", loss_q)
        self.log("train/generator/loss", loss_g)
        opt_g.step()
        sch_g.step()
    def validation_step(self,batch,batch_idx):
        wav, speaker,wav_lens,filename = batch["resampled_speech.pth"], batch['speaker'], batch['wav_lens'], batch['filenames']
        with torch.no_grad():
            wav_mel = self.calc_logmelspec(wav).detach()

        c = self.encoder(wav.unsqueeze(1))

        q, loss_q, c = self.quantizer(c)
        speaker = self.speaker_embedding(speaker)
        wav_generator_out = self.generator(q,speaker)
        predicted_mel = self.calc_logmelspec(wav_generator_out.squeeze(1))
        min_len = min(predicted_mel.size(2), wav_mel.size(2))
        loss_recons = self.reconstruction_loss(wav_mel[:,:,:min_len], predicted_mel[:,:,:min_len])
        if (
            batch_idx < self.cfg.model.quantizer.logging_wav_samples
            and self.global_rank == 0
            and self.local_rank == 0
        ):
            self.log_audio(
                wav_generator_out[0]
                .squeeze()[: wav_lens[0]]
                .float()
                .cpu()
                .numpy()
                .astype(np.float32),
                name=f"generated/{filename[0]}",
                sampling_rate=self.cfg.sample_rate,
            )
            self.log_audio(
                wav[0].squeeze()[: wav_lens[0]].cpu().float().numpy().astype(np.float32),
                name=f"natural/{filename[0]}",
                sampling_rate=self.cfg.sample_rate,
            )

        self.log("val/reconstruction", loss_recons)
        self.log("val/quantization", loss_q)


    def configure_optimizers(self):
        opt_g = hydra.utils.instantiate(
            self.cfg.model.quantizer.optim.opt_g, params=
            itertools.chain(self.generator.parameters(),
                            self.encoder.parameters(),
                            self.quantizer.parameters(),
                            self.speaker_embedding.parameters())
        )
        opt_d = hydra.utils.instantiate(
            self.cfg.model.quantizer.optim.opt_d,
            params=itertools.chain(
                self.msd.parameters(),
                self.mpd.parameters(),
            ),
        )
        scheduler_g = hydra.utils.instantiate(
            self.cfg.model.quantizer.optim.scheduler_g, optimizer=opt_g
        )
        scheduler_d = hydra.utils.instantiate(
            self.cfg.model.quantizer.optim.scheduler_d, optimizer=opt_d
        )

        return [opt_g, opt_d], [
            {"name": "scheduler_g", "scheduler": scheduler_g},
            {"name": "scheduler_d", "scheduler": scheduler_d},
        ]
    def calc_logmelspec(self,x,min_clip_val=1e-5):
        mel = self.mel_spec(x)
        logmel = torch.log(torch.clamp_min(mel,min=min_clip_val))
        return logmel
    def log_audio(self, audio, name, sampling_rate):
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
