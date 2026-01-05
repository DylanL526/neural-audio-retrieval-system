import tensorflow as tf
import tensorflow_hub as hub
import faiss
import numpy as np
import librosa
from pathlib import Path

# Load VGGish neural network
model = hub.load('https://www.kaggle.com/models/google/vggish/TensorFlow2/vggish/1')
db_vectors = []
db_filenames = []
file_paths = list(Path('./data').glob('*.wav'))

# Build vectors by using VGGish embedding model
def build_database(file_paths):
    for file_path in file_paths:
        try:
            y, sr = librosa.load(file_path, sr=16000, mono=True)
            y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
            embeddings = model(y_tensor)
            vector = tf.reduce_mean(embeddings, axis=0)
            db_vectors.append(vector.numpy())
            db_filenames.append(file_path.name)
        except:
            print(f'Failed to load {file_path}\n')

build_database(file_paths)

# Create vector database for retrieval purposes with FAISS
if len(db_vectors) > 0:
    db_vectors = np.array(db_vectors).astype('float32')
    dimension = 128
    index = faiss.IndexFlatL2(dimension)
    index.add(db_vectors)
else:
    print('No database vectors found.')
    exit()

# Perform similarity search based on user input audio file name
def search_database():
    user_input = input('Enter an audio file path to search for similar audio: ')
    user_input = user_input.strip()
    query_path = Path(user_input)
    if query_path.exists():
        y, sr = librosa.load(query_path, sr=16000, mono=True)
        y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
        embeddings = model(y_tensor)
        query_vector = tf.reduce_mean(embeddings, axis=0)
        query_matrix = query_vector.numpy()[np.newaxis, :].astype('float32')
        
        # Search for similar audio
        k = min(6, len(db_vectors))
        distances, indices = index.search(query_matrix, k)
        
        start_index = 1
        print(f'\nMost similar audio:')
        if distances[0][0] > 0:
            start_index = 0
        for i in range(start_index, start_index + k - 1):
            idx = indices[0][i]
            dist = distances[0][i]
            filename = db_filenames[idx]
            print(f'{i}. {filename} (Distance: {dist})')
    else:
        print('File does not exist. Please try again.')
        search_database()

search_database()