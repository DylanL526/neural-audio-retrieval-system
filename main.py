import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

model = hub.load('https://www.kaggle.com/models/google/vggish/TensorFlow2/vggish/1')

file_path = librosa.ex('trumpet')
y, sr = librosa.load(file_path)

spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('This is what an AI "sees" when it listens to a Trumpet')
plt.tight_layout()
plt.show()