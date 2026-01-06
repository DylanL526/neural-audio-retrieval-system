# Neural Audio Retrieval System

An audio retrieval pipeline utilizing Google's [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish) deep neural network and Meta's [FAISS](https://github.com/facebookresearch/faiss)
to embed and search for mathematically similar audio. As opposed to searching for semantically similar audio based on names or descriptions, this engine analyzes the raw audio waveform,
extracts semantic features, and retrieves other audio files based on a kNN search. Applications include implementations in music recommendation systems or automated audio copyright detection.
