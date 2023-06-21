import torch
from torch.utils.data.dataset import Dataset
import torchaudio
from pathlib import Path
import random
import string


class ParallelAudiobookCorpus(Dataset):
    def __init__(self,root) -> None:
        super().__init__()
        self.root = Path(root)
        self.books = [f.stem for f in self.root.glob("*") if f.is_dir()]
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
    def __getitem__(self, index):
        wav_path = self.wav_files[index]
        wav_tensor,sr = torchaudio.load(wav_path)
        wav_path = wav_path.resolve()
        speaker = wav_path.parent.stem
        book = wav_path.parent.parent.parent.stem
        utt_id = wav_path.stem

        clean_text = self.clean_texts[f"{book}_{utt_id}"]
        punc_text = self.punc_texts[f"{book}_{utt_id}"]

        return wav_tensor,sr,str(wav_path),speaker,book,utt_id,clean_text,punc_text
    def __len__(self):
        return len(self.wav_files)
        
        
