## Extracting incremental HuBERT embeddings

`extract_incremental_embeddings.py` contains functions to extract frame-by-frame HuBERT embeddings in which each frame reflects only the audio up to that point.

Run `emb = extract_hubert_embeddings("path/to/audio.wav")` on each audio file of interest. 

Each call saves `hubert_embeddings_incremental/<filename>_embeddings.pt`, a dict mapping hidden-state index (0 = input to the first transformer layer, 1–48 = transformer layers) to a tensor of shape (n_frames, 1280), one row per 20 ms frame. If that file already exists, it is loaded instead of recomputed.
