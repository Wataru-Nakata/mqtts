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
        with open(cfg.data.vocab_path) as f:
            self.vocab = {p.replace('\n',''):idx for idx,p in enumerate(f.readlines())}


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
            collate_fn=self.collate_fn,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.val_batch_size,
            collate_fn=self.collate_fn,
            num_workers=8,
        )


    @torch.no_grad()
    def collate_fn(self, batch, segment_size: int = -1):

        outputs = dict()
        outputs["resampled_speech.pth"] = pad_sequence(
            [b["resampled_speech.pth"].squeeze() for b in batch], batch_first=True
        )
        outputs['tokens'] = pad_sequence([torch.stack(b["quantized_code.pth"],dim=0).T for b in batch], batch_first=True)
        outputs['tokens_mask'] = self.length_to_mask(torch.tensor([len(b['quantized_code.pth'][0]) for b in batch])+1)
        outputs['phones'] = pad_sequence([torch.tensor([self.vocab[p] for p in b["phones.txt"].split("|")]) for b in batch],batch_first=True)
        outputs['phones_mask'] =  self.length_to_mask(torch.tensor([len(b['phones.txt'].split("|")) for b in batch]) )

        start = torch.ones((len(batch),1,4),dtype=torch.int)*(self.cfg.model.quantizer.quantizer.n_codes+1)
        end = torch.ones((1,4),dtype=torch.int)*(self.cfg.model.quantizer.quantizer.n_codes+2)

        outputs['input_tokens'] = torch.cat([start,outputs['tokens']],dim=1)
        end_added = [torch.cat([torch.stack(b["quantized_code.pth"]).T,end],dim=0) for b in batch]
        outputs['output_tokens'] = pad_sequence(end_added,batch_first=True)

        outputs['input_tokens'] = outputs['input_tokens'][:,:2000,:]
        outputs['output_tokens'] = outputs['output_tokens'][:,:2000,:]
        outputs['tokens_mask'] = outputs['tokens_mask'][:,:2000]


        
        outputs["wav_lens"] = torch.tensor(
            [b["resampled_speech.pth"].size(0) for b in batch]
        )

        outputs["filenames"] = [b["__key__"] for b in batch]
        outputs["speaker"] = torch.tensor([self.speaker_dict[b["speaker.txt"]] for b in batch])
        return outputs
    def length_to_mask(self,length, max_len=None, dtype=None):
        """length: B.
        return B x max_len.
        If max_len is None, then max of length will be used.
        """
        assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
        max_len = max_len or length.max().item()
        mask = torch.arange(max_len, device=length.device,
                            dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
        if dtype is not None:
            mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
        return ~mask
