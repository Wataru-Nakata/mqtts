import webdataset as wds
import lightning
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import torch
from torch.utils.data import random_split
import json
import math
import random
import torchaudio
import hydra

class MQTTSDataModule(lightning.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        with open(cfg.data.speaker_dict) as f:
            self.speaker_dict = json.load(f)

    def setup(self, stage: str):
        self.train_dataset = (
            wds.WebDataset(self.cfg.data.train_dataset_path)
            .shuffle(1000)
            .decode(wds.torch_audio)
        )
        self.val_dataset = wds.WebDataset(self.cfg.data.val_dataset_path).decode(
            wds.torch_audio
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.train_batch_size,
            collate_fn=lambda batch: self.collate_fn(
                batch, self.cfg.data.train_segment_size
            ),
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.val_batch_size,
            collate_fn=lambda batch: self.collate_fn(
                batch, -1
            ),
            num_workers=8,
        )


    @torch.no_grad()
    def collate_fn(self, batch, segment_size: int = -1):

        outputs = dict()
        if segment_size != -1:
            cropped_speeches = []
            for sample in batch:
                wav = sample["resampled_speech.pth"]
                feature_len = wav.size(0)
                if feature_len > (segment_size+1):
                    feature_start = random.randint(
                        0, feature_len - segment_size - 1
                    )
                    feature_end = segment_size + feature_start
                    cropped_speeches.append(
                        wav.squeeze()[
                            int(feature_start) : int(feature_end)
                        ]
                    )
                else:
                    cropped_speeches.append(wav.squeeze())
            outputs["resampled_speech.pth"] = pad_sequence(
                cropped_speeches, batch_first=True
            )
        else:
            outputs["resampled_speech.pth"] = pad_sequence(
                [b["resampled_speech.pth"].squeeze() for b in batch], batch_first=True
            )
        
        outputs["wav_lens"] = torch.tensor(
            [b["resampled_speech.pth"].size(0) for b in batch]
        )

        outputs["filenames"] = [b["__key__"] for b in batch]
        outputs["speaker"] = torch.tensor([self.speaker_dict[b["speaker"].decode()] for b in batch])
        return outputs
