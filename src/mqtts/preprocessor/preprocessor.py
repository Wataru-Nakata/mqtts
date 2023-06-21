import torch
import hydra
import torchaudio
import pathlib
from omegaconf import DictConfig
import numpy as np
import webdataset
import tqdm
from torch.utils.data import DataLoader


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
        self.dataset = hydra.utils.instantiate(cfg.preprocess.preprocess_dataset)
        self.sampling_rate = self.cfg.sample_rate

    @torch.no_grad()
    def process_utterance(
        self,
        orig_waveform: torch.Tensor,
        sample_rate: int,
        audio_file_path,
        speaker:str,
        book:str,
        utt_id:str,
        clean_text:str,
        punc_text:str
    ):
        waveform = torchaudio.functional.resample(
            orig_waveform, sample_rate, new_freq=self.sampling_rate
        )[
            0
        ]  # remove channel dimension only support mono

        with open(audio_file_path, mode="rb") as f:
            wav_bytes = f.read()
        basename= f"{book}_{speaker}_{utt_id}"

        sample = {
            "__key__": basename,
            "speech.wav": wav_bytes,
            "resampled_speech.pth": webdataset.torch_dumps(waveform),
            "clean_text": clean_text,
            "punc_text": punc_text,
            "speaker": speaker,
            "book": book,
            "utt_id": utt_id
        }
        return sample

    def build_from_path(self):
        pathlib.Path("/".join(self.cfg.preprocess.train_tar_sink.pattern.split("/")[:-1])).mkdir(exist_ok=True)
        train_sink = hydra.utils.instantiate(self.cfg.preprocess.train_tar_sink)
        val_sink = hydra.utils.instantiate(self.cfg.preprocess.val_tar_sink)
        dataloader = DataLoader(self.dataset,batch_size=1)
        for idx, (wav_tensor,sr,wav_path,speaker,book,utt_id,clean_text,punc_text) in enumerate(tqdm.tqdm(dataloader)):
            sample = self.process_utterance(
                wav_tensor[0],sr[0],wav_path[0],speaker[0],book[0],utt_id[0],clean_text[0],punc_text[0]
            )
            if idx >= self.cfg.preprocess.val_size:
                train_sink.write(sample)
            else:
                val_sink.write(sample)

        train_sink.close()
        val_sink.close()

