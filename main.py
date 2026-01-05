import tensorflow as tf
import tensorflow_hub as hub
import faiss
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

model = hub.load('https://www.kaggle.com/models/google/vggish/TensorFlow2/vggish/1')
db_vectors = []
db_filenames = []
file_paths = Path('./data').glob('*.wav')

def build_database(file_paths):
    for file_path in file_paths:
        y, sr = librosa.load(file_path, sr=16000, mono=True)
        y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
        embeddings = model(y_tensor)
        vector = tf.reduce_mean(embeddings, axis=0)
        db_vectors.append(vector.numpy())
        db_filenames.append(file_path.name)

build_database(file_paths)

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