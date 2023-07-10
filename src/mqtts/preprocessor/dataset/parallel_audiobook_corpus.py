import phonemizer
from phonemizer.backend import EspeakBackend
import torch
from torch.utils.data.dataset import Dataset
import torchaudio
from pathlib import Path
import random
import string
from phonemizer import phonemize




class ParallelAudiobookCorpus(Dataset):
    def __init__(self,root,exclude_books=[]) -> None:
        super().__init__()
        self.root = Path(root)
        self.books = [f.stem for f in self.root.glob("*") if f.is_dir() and f.stem not in exclude_books]
        self.clean_texts = dict()
        self.punc_texts = dict()
        for book in self.books:
            with (self.root/book/"txt.clean").open() as f:
                lines = f.readlines()
                utt_ids, texts = zip(*[l.strip().split("\t") for l in lines])
                for utt_id, text in zip(utt_ids,texts):
                    self.clean_texts[f'{book}_{utt_id}'] = text
            with (self.root/book/"txt.punc").open() as f:
                lines = f.readlines()
                utt_ids, texts = zip(*[l.strip().split("\t") for l in lines])
                for utt_id, text in zip(utt_ids,texts):
                    self.punc_texts[f'{book}_{utt_id}'] = text
        self.wav_files = list(self.root.glob("**/ch*_utt*.wav"))
        self.phonemizer = EspeakBackend('en-us',preserve_punctuation=True)
    def __getitem__(self, index):
        wav_path = self.wav_files[index]
        wav_tensor,sr = torchaudio.load(wav_path)
        wav_path = wav_path.resolve()
        speaker = wav_path.parent.stem
        book = wav_path.parent.parent.parent.stem
        utt_id = wav_path.stem

        clean_text = self.clean_texts[f"{book}_{utt_id}"].strip()
        punc_text = self.punc_texts[f"{book}_{utt_id}"].strip()
        basename= f"{book}_{speaker}_{utt_id}"
        output = {
            "wav_tensor": wav_tensor,
            "sr": sr,
            "wav_path": str(wav_path),
            "speaker": speaker,
            "book": book,
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
        for spk_file in self.root.glob("**/spk2utt"):
            with spk_file.open() as f:
                lines = f.readlines()
                [speakers.add(l.split()[0]) for l in lines]
        speaker_dict = {x:idx for idx,x in enumerate(speakers)}
        return speaker_dict
    def phonemize(self,lines):
        return [list(l) for l in self.phonemizer.phonemize(lines,strip=True)]

