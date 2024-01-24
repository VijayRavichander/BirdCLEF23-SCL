import librosa as lb
import librosa.display as lbd
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config

class BirdDataset(Dataset):
    def __init__(self, data, sr=config.SR, n_mels=128, fmin=0, fmax=None, duration=config.DURATION, step=None, res_type="kaiser_fast", resample=True):
        
        self.data = data
        
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr//2

        self.duration = duration
        self.audio_length = self.duration*self.sr
        self.step = step or self.audio_length
        
        self.res_type = res_type
        self.resample = resample

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def normalize(image):
        image = image.astype("float32", copy=False) / 255.0
        image = np.stack([image, image, image])
        return image
    
    def compute_melspec(self, y, sr, n_mels, fmin, fmax):

        melspec = lb.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax,
        )

        melspec = lb.power_to_db(melspec).astype(np.float32)
        return melspec

    def mono_to_color(self, X, eps=1e-6, mean=None, std=None):
        mean = mean or X.mean()
        std = std or X.std()
        X = (X - mean) / (std + eps)
        
        _min, _max = X.min(), X.max()

        if (_max - _min) > eps:
            V = np.clip(X, _min, _max)
            V = 255 * (V - _min) / (_max - _min)
            V = V.astype(np.uint8)
        else:
            V = np.zeros_like(X, dtype=np.uint8)

        return V

    def crop_or_pad(self, y, length, is_train=True, start=None):
        if len(y) < length:
            y = np.concatenate([y, np.zeros(length - len(y))])
            
            n_repeats = length // len(y)
            epsilon = length % len(y)
            
            y = np.concatenate([y]*n_repeats + [y[:epsilon]])
            
        elif len(y) > length:
            if not is_train:
                start = start or 0
            else:
                start = start or np.random.randint(len(y) - length)

            y = y[start:start + length]

        return y
    
    def audio_to_image(self, audio):
        melspec = self.compute_melspec(audio, self.sr, self.n_mels, self.fmin, self.fmax) 
        image = self.mono_to_color(melspec)
        image = self.normalize(image)
        return image

    def read_file(self, filepath):
        audio, orig_sr = sf.read(filepath, dtype="float32")

        if self.resample and orig_sr != self.sr:
            audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)
          
        audios = []
        
        #converting the entire audio into 5 secs chunks
        for i in range(self.audio_length, len(audio) + self.step, self.step):
            start = max(0, i - self.audio_length)
            end = start + self.audio_length
            audios.append(audio[start:end])
            
        if len(audios[-1]) < self.audio_length:
            audios = audios[:-1]
            
        images = [self.audio_to_image(audio) for audio in audios]
        images = np.stack(images)
        
        return images
    
        
    def __getitem__(self, idx):
        return self.read_file(self.data.loc[idx, "path"])