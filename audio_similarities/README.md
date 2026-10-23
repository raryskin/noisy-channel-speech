## Extracting incremental HuBERT embeddings

`extract_incremental_embeddings.py` contains functions to extract frame-by-frame HuBERT embeddings in which each frame reflects only the audio up to that point.

Run `emb = extract_hubert_embeddings("path/to/audio.wav")` on each audio file of interest. 

Each call saves `hubert_embeddings_incremental/<filename>_embeddings.pt`, a dict mapping hidden-state index (0 = input to the first transformer layer, 1–48 = transformer layers) to a tensor of shape (n_frames, 1280), one row per 20 ms frame. If that file already exists, it is loaded instead of recomputed.

## Computing DTW similarities

`extract_DTW_sim.py` contains functions to align the incremental embeddings of each pair of audio files with dynamic time warping (DTW) and compute the cosine similarity at every point on the alignment path, separately for each HuBERT layer. After running it, run ```run_pair_list(pair_csv="path/to/pairs.csv", embedding_dir="path/to/embeddings/", output_dir="path/to/dtw_output/")```

### Input

A CSV listing the pairs to compare, with columns `fileA` and `fileB` containing paths to the audio files. `fileA` is the reference file (in the paper, the target word), and `fileB` is the file it is compared against (a competitor or distractor). 


### Output

For each pair and layer, the following are written to `output_dir`:

- `<A>__<B>__layer<L>_sim_matrix.npy`: cosine similarity between every frame of
  A and every frame of B (frames of A × frames of B)
- `<A>__<B>__layer<L>_path.npy`: the DTW alignment path as (A frame, B frame)
  index pairs
- `<A>__<B>__layer<L>_similarities.npy`: cosine similarity at each step of the path
- `<A>__<B>__layer<L>_timeseries_A.csv` and `_timeseries_B.csv`: the path
  similarities as a timeseries, with `time` given by the frame index of A or B
  respectively (frame index × 0.02 s). The manuscript uses only the timeseries anchored to the target file
(`_timeseries_A.csv`)
- `<A>__<B>_dtw_summary.csv`: mean, median, min and max path similarity and
  path length per layer

A combined `all_pairs_dtw_summary.csv` covering every pair is also written.
