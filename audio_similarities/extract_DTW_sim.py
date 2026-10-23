import numpy as np
import pandas as pd
import torch

from pathlib import Path

import librosa
from librosa.sequence import dtw


def cosine_similarity_matrix(emb1, emb2):

    emb1 = torch.nn.functional.normalize(
        emb1,
        dim=-1
    )

    emb2 = torch.nn.functional.normalize(
        emb2,
        dim=-1
    )

    sim = emb1 @ emb2.T

    return sim.cpu().numpy()
  

def dtw_alignment_for_layer(
    emb1,
    emb2
):

    # cosine similarity matrix
    sim = cosine_similarity_matrix(
        emb1,
        emb2
    )

    # DTW minimizes cost
    cost = 1.0 - sim
    
    accumulated_cost, path = dtw(C=cost)

    # librosa returns path backwards
    path = path[::-1]

    # cosine similarities along DTW path
    path_sims = sim[
        path[:, 0],
        path[:, 1]
    ]

    path_length = len(path)

    max_len = max(
        emb1.shape[0],
        emb2.shape[0]
    )

    results = {
        "similarity_matrix": sim,
        "path": path,
        "path_similarities": path_sims,
        "mean_similarity": np.mean(path_sims),
        "median_similarity": np.median(path_sims),
        "max_similarity": np.max(path_sims),
        "min_similarity": np.min(path_sims),
        "path_length": path_length,
        "path_ratio": path_length / max_len
    }

    return results

def dtw_to_timeseries(
    path,
    path_sims,
    embA,
    embB,
    durationA=None,
    durationB=None,
    anchor="A"   # "A" or "B"
):
    """
    Converts DTW path into a time-aligned similarity series
    anchored to either fileA or fileB.
    """

    path = np.array(path)

    i_idx = path[:, 0]
    j_idx = path[:, 1]

    if anchor == "A":
        idx = i_idx
        duration = durationA
        n_frames = embA.shape[0]

    elif anchor == "B":
        idx = j_idx
        duration = durationB
        n_frames = embB.shape[0]

    else:
        raise ValueError("anchor must be 'A' or 'B'")

    # convert frame index → time
    FRAME_STEP = 0.02  # 20 ms per frame (typical HuBERT rate)

    t = idx * FRAME_STEP

    return pd.DataFrame({
        "time": t,
        "similarity": path_sims,
        "A_idx": i_idx,
        "B_idx": j_idx
    })
    
    
def compare_pair_dtw(
    fileA,
    fileB,
    embedding_dir,
    output_dir
):

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    embA = torch.load(
        Path(embedding_dir) / f"{fileA}.pt",
        map_location="cpu"
    )

    embB = torch.load(
        Path(embedding_dir) / f"{fileB}.pt",
        map_location="cpu"
    )

    summary_rows = []

    for layer in sorted(embA.keys()):

        print(
            f"{fileA} vs {fileB} layer {layer}"
        )
        
        result = dtw_alignment_for_layer(
                embA[layer],
                embB[layer]
            )
            
            # DTW to timeseries
        ts_A = dtw_to_timeseries(
            result["path"],
            result["path_similarities"],
            embA[layer],
            embB[layer],
            anchor="A"
        )
        
        ts_B = dtw_to_timeseries(
            result["path"],
            result["path_similarities"],
            embA[layer],
            embB[layer],
            anchor="B"
        )    

        # --------------------
        # save DTW path
        # --------------------

        np.save(
            output_dir /
            f"{fileA}__{fileB}__layer{layer}_path.npy",
            result["path"]
        )

        # --------------------
        # save similarity trace
        # --------------------

        np.save(
            output_dir /
            f"{fileA}__{fileB}__layer{layer}_similarities.npy",
            result["path_similarities"]
        )
        
        # --------------------
        # save similarity matrix
        # --------------------

        np.save(
            output_dir /
            f"{fileA}__{fileB}__layer{layer}_sim_matrix.npy",
            result["similarity_matrix"]
        )
        
        # -----
        # save timeseries anchored to file A or B
        # -----
        
        ts_A.to_csv(
            output_dir /
            f"{fileA}__{fileB}__layer{layer}_timeseries_A.csv",
            index=False
        )
        
        ts_B.to_csv(
            output_dir /
            f"{fileA}__{fileB}__layer{layer}_timeseries_B.csv",
            index=False
        )
        
        # --------------------
        # summary row
        # --------------------

        summary_rows.append(
            {
                "fileA": fileA,
                "fileB": fileB,
                "layer": layer,
                "mean_similarity":
                    result["mean_similarity"],
                "median_similarity":
                    result["median_similarity"],
                "max_similarity":
                    result["max_similarity"],
                "min_similarity":
                    result["min_similarity"],
                "path_length":
                    result["path_length"],
                "path_ratio":
                    result["path_ratio"]
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        output_dir /
        f"{fileA}__{fileB}_dtw_summary.csv",
        index=False
    )

    return summary_df
  

def run_pair_list(
    pair_csv,
    embedding_dir,
    output_dir
):

    pairs = pd.read_csv(pair_csv)

    all_results = []

    for _, row in pairs.iterrows():

        # Remove path and .wav extension,
        # then append "_embeddings"

        fileA = (
            Path(row["fileA"]).stem
            + "_embeddings"
        )

        fileB = (
            Path(row["fileB"]).stem
            + "_embeddings"
        )

        print(f"Comparing {fileA} vs {fileB}")

        summary = compare_pair_dtw(
            fileA,
            fileB,
            embedding_dir,
            output_dir
        )

        all_results.append(summary)

    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    combined.to_csv(
        Path(output_dir) /
        "all_pairs_dtw_summary.csv",
        index=False
    )

    return combined
