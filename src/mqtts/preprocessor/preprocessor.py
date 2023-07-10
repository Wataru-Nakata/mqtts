from statistics import mode
import torch
import hydra
import torchaudio
import pathlib
from omegaconf import DictConfig
import numpy as np
from torchmetrics import audio
import webdataset
import tqdm
from torch.utils.data import DataLoader
import json
from phonemizer import phonemize

from mqtts.model.quantizer_lightning_module import QuantizerLightningModule


class Preprocessor:
    """
    Preprocess dataset
    """

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: hydra config
        """
        self.cfg = cfg
        self.dataset = hydra.utils.instantiate(cfg.data.dataset)
        self.sampling_rate = self.cfg.sample_rate
        self.speaker_set = set()
        self.vocab_list = list()
        self.vocab_list.append("<pad>")
        self.quantizer = QuantizerLightningModule.load_from_checkpoint(self.cfg.data.quantizer_path,cfg=cfg)

    @torch.no_grad()
    def process_utterance(
        self,
        sample
    ):
        orig_waveform = sample['wav_tensor']
        sample_rate= sample['sr']
        audio_file_path = sample['wav_path']
        speaker = sample["speaker"]
        basename=sample["basename"]
        clean_text = sample['clean_text']
        waveform = torchaudio.functional.resample(
            orig_waveform, sample_rate, new_freq=self.sampling_rate
        )[
            0
        ]  # remove channel dimension only support mono

        with open(audio_file_path, mode="rb") as f:
            wav_bytes = f.read()
        latent,_,codes = self.quantizer(waveform.view(1,-1).to(self.quantizer.device))
        latent = latent.cpu()
        codes = [c.cpu()for c in codes]
        phones = self.dataset.phonemize([clean_text])[0]
        for p in phones:
            if p not in self.vocab_list:
                self.vocab_list.append(p)
        if type(phones) == list:
            phones = "|".join(phones)

        sample = {
            "__key__": basename,
            "speech.wav": wav_bytes,
            "resampled_speech.pth": webdataset.torch_dumps(waveform),
            "clean.txt": clean_text,
            "phones.txt": phones,
            "quantized_code.pth": webdataset.torch_dumps(codes),
            "speaker.txt": speaker
        }
        for k, v in sample.items():
            if k in ['wav_tensor', 'sr', 'wav_path', 'speaker','basename',"clean_text"]:
                continue
            else:
                sample[k] = v

        self.speaker_set.add(speaker)
        return sample

    def build_from_path(self):
        pathlib.Path("/".join(self.cfg.data.train_tar_sink.pattern.split("/")[:-1])).mkdir(exist_ok=True)
        train_sink = hydra.utils.instantiate(self.cfg.data.train_tar_sink)
        val_sink = hydra.utils.instantiate(self.cfg.data.val_tar_sink)
        dataloader = DataLoader(self.dataset,batch_size=1,shuffle=True)
        for idx, sample in enumerate(tqdm.tqdm(dataloader)):
            sample = self.process_utterance({k:v[0] for k,v in sample.items()})
            if idx >= self.cfg.data.val_size:
                train_sink.write(sample)
            else:
                val_sink.write(sample)
        with open(self.cfg.data.speaker_dict,mode='w') as f:
            speaker_dict = {x:idx for idx,x in enumerate(self.speaker_set)}
            f.write(json.dumps(speaker_dict))
        with open(self.cfg.data.vocab_path,mode='w') as f:
            f.writelines("\n".join(self.vocab_list))

        train_sink.close()
        val_sink.close()

