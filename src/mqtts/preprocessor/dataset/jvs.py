import phonemizer
from phonemizer.backend import EspeakBackend
import torch
from torch.utils.data.dataset import Dataset
import torchaudio
from pathlib import Path
import random
import string
import pyopenjtalk
import re

def numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))

def pp_symbols(labels, drop_unvoiced_vowels=True):
    """Extract phoneme + prosoody symbol sequence from input full-context labels

    The algorithm is based on [Kurihara 2021] [1]_ with some tweaks.

    Args:
        labels (HTSLabelFile): List of labels
        drop_unvoiced_vowels (bool): Drop unvoiced vowels. Defaults to True.

    Returns:
        list: List of phoneme + prosody symbols

    .. ipython::

        In [11]: import ttslearn

        In [12]: from nnmnkwii.io import hts

        In [13]: from ttslearn.tacotron.frontend.openjtalk import pp_symbols

        In [14]: labels = hts.load(ttslearn.util.example_label_file())

        In [15]: " ".join(pp_symbols(labels.contexts))
        Out[15]: '^ m i [ z u o # m a [ r e ] e sh i a k a r a ... $'

    .. [1] K. Kurihara, N. Seiyama, and T. Kumano, “Prosodic features control by
        symbols as input of sequence-to-sequence acoustic modeling for neural tts,”
        IEICE Transactions on Information and Systems, vol. E104.D, no. 2,
        pp. 302–311, 2021.
    """
    PP = []
    N = len(labels)

    # 各音素毎に順番に処理
    for n in range(N):
        lab_curr = labels[n]

        # 当該音素
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)  # type: ignore

        # 無声化母音を通常の母音として扱う
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # 先頭と末尾の sil のみ例外対応
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                PP.append("^")
            elif n == N - 1:
                # 疑問系かどうか
                e3 = numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    PP.append("$")
                elif e3 == 1:
                    PP.append("?")
            continue
        elif p3 == "pau":
            PP.append("_")
            continue
        else:
            PP.append(p3)

        # アクセント型および位置情報（前方または後方）
        a1 = numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
        # アクセント句におけるモーラ数
        f1 = numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])

        # アクセント句境界
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            PP.append("#")
        # ピッチの立ち下がり（アクセント核）
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            PP.append("]")
        # ピッチの立ち上がり
        elif a2 == 1 and a2_next == 2:
            PP.append("[")

    return PP



class JVSCorpus(Dataset):
    def __init__(self,root,exclude_speakers=[]) -> None:
        super().__init__()
        self.root = Path(root)
        self.speakers = [f.stem for f in self.root.glob("jvs*") if f.is_dir() and f.stem not in exclude_speakers]
        self.clean_texts = dict()
        self.wav_files  = []
        for speaker in self.speakers:
            transcript_files = (self.root/speaker).glob("**/transcripts_utf8.txt")
            for transcript_file in transcript_files:
                subset = transcript_file.parent.name
                with transcript_file.open() as f:
                    lines = f.readlines()
                for line in lines:
                    wav_name, text = line.strip().split(":")
                    self.clean_texts[f"{speaker}/{subset}/{wav_name}"] =text
                    wav_path = self.root/ Path(f"{speaker}/{subset}/wav24kHz16bit/{wav_name}.wav")
                    if  wav_path.exists():
                        self.wav_files.append(wav_path)
            
    def __getitem__(self, index):
        wav_path = self.wav_files[index]
        wav_tensor,sr = torchaudio.load(wav_path)
        wav_path = wav_path.resolve()
        speaker = wav_path.parent.parent.parent.stem
        subset = wav_path.parent.parent.stem
        wav_name = wav_path.stem

        clean_text = self.clean_texts[f"{speaker}/{subset}/{wav_name}"]
        basename= f"{subset}_{speaker}_{wav_name}"
        output = {
            "wav_tensor": wav_tensor,
            "sr": sr,
            "wav_path": str(wav_path),
            "speaker": speaker,
            "clean_text": clean_text,
            "basename": basename
        #    "phones": phones
        }

        return output
    def __len__(self):
        return len(self.wav_files)
    
    @property
    def speaker_dict(self):
        speakers = set()
        for wav_path in self.wav_files:
            speakers.add(wav_path.parent.parent.parent.stem)
        speaker_dict = {x:idx for idx,x in enumerate(speakers)}
        return speaker_dict
    def phonemize(self,lines):
        full_context = pyopenjtalk.extract_fullcontext(lines[0])
        return [pp_symbols(full_context)]


