from os.path import basename
import phonemizer
from phonemizer.backend import EspeakBackend
import torch
from torch.utils.data.dataset import Dataset
import torchaudio
from pathlib import Path
import random
import string
from phonemizer import phonemize
import re



class LibriTTSCorpus(Dataset):
    def __init__(self,root) -> None:
        super().__init__()
        self.root = Path(root)
        self.wav_files = list(self.root.glob("**/*.wav"))
        self.wav_files = [x for x in self.wav_files if (Path(str(x).replace(".wav",".normalized.txt")).exists() and Path(str(x).replace(".wav",".original.txt")).exists())]
        self.phonemizer = EspeakBackend('en-us',preserve_punctuation=True)
    def __getitem__(self, index):
        wav_path = self.wav_files[index]
        wav_tensor,sr = torchaudio.load(wav_path)
        wav_path = wav_path.resolve()
        basename = wav_path.stem
        m = re.search(r"^(\d+?)\_(\d+?)\_(\d+?\_\d+?)$", basename)
        speaker,chapter, utt_id = m.group(1), m.group(2), m.group(3)
        with wav_path.with_suffix(".normalized.txt").open() as f:
            lines = f.readlines()
            line = " ".join(lines)
            line = line.strip()
        clean_text = line
        with wav_path.with_suffix(".original.txt").open() as f:
            lines = f.readlines()
            line = " ".join(lines)
            line = line.strip()
        punc_text = line

        output = {
            "wav_tensor": wav_tensor,
            "sr": sr,
            "wav_path": str(wav_path),
            "speaker": speaker,
            "chapter": chapter,
            "utt_id": utt_id,
            "clean_text": clean_text,
            "punc_text": punc_text,
            "basename": basename
        #    "phones": phones
        }

        return output
    def __len__(self):
        return len(self.wav_files)
    
    @property
    def speaker_dict(self):
        speakers = set()
        for wav_file in self.wav_files:
            basename = wav_file.stem
            m = re.search(r"^(\d+?)\_(\d+?)\_(\d+?\_\d+?)$", basename)
            speaker,chapter, utt_id = m.group(1), m.group(2), m.group(3)
            speakers.add(speaker)
        speaker_dict = {x:idx for idx,x in enumerate(speakers)}
        return speaker_dict
    def phonemize(self,lines):
        return [list(l) for l in self.phonemizer.phonemize(lines,strip=True)]

