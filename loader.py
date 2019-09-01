import numpy as np
import scipy.io.wavfile as wav
import os

class DataLoader():
    def __init__(self, dataset):
        self.dataset = dataset
        self.train_inputs = []
        self.train_labels = []
        self.validation_inputs = []
        self.validation_labels = []
        self.test_inputs = []
        self.test_labels = []

    def load(self):
        if self.dataset == 'fsdd':
            root_dir = 'free-spoken-digit-dataset/recordings/'
            for file_name in sorted(os.listdir(root_dir)):
                fs, raw_wav = wav.read(root_dir+file_name)

                max_volume = np.amax(np.abs(raw_wav))
                raw_wav = raw_wav * 32767 / max_volume

                label, speaker, index = file_name.split('.', 1)[0].split('_')
                label = int(label)
                index = int(index)

                if index < 5:
                    self.validation_inputs.append(raw_wav)
                    self.validation_labels.append(label)
                else:
                    self.train_inputs.append(raw_wav)
                    self.train_labels.append(label)

            self.test_inputs = self.validation_inputs
            self.test_labels = self.validation_labels
