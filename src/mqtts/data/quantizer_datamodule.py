import webdataset as wds
import lightning
from omegaconf import DictConfig
from torch.utils.data import DataLoader,random_split
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import torch
from torch.utils.data import random_split
import json
import math
import random
import torchaudio
import hydra

class MQTTSQuantizerDataModule(lightning.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str):
        dataset = hydra.utils.instantiate(self.cfg.data.dataset)

        self.train_dataset,self.val_dataset = random_split(dataset,[0.9,0.1])
        self.speaker_dict = dataset.speaker_dict
        print(self.speaker_dict)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.train_batch_size,
            collate_fn=lambda batch: self.collate_fn(
                batch, self.cfg.data.train_segment_size
            ),
            num_workers=16,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.val_batch_size,
            collate_fn=lambda batch: self.collate_fn(
                batch, -1
            ),
            num_workers=16,
        )


    @torch.no_grad()
    def collate_fn(self, batch, segment_size: int = -1):
        for i in range(len(batch)):
            sample = batch[i]
            if sample['sr'] != self.cfg.sample_rate:
                resampled = torchaudio.functional.resample(sample['wav_tensor'],sample['sr'],self.cfg.sample_rate)
                batch[i]['resampled_speech.pth'] = resampled.squeeze()
            else:
                batch[i]['resampled_speech.pth'] = sample['wav_tensor'].squeeze()
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

        outputs["filenames"] = [b["basename"] for b in batch]
        outputs["speaker"] = torch.tensor([self.speaker_dict[b["speaker"]] for b in batch])
        return outputs
