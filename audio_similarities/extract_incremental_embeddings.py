import torch
from pathlib import Path
from torchcodec.decoders import AudioDecoder
from transformers import Wav2Vec2Processor, HubertModel

# ----------------------------
# SETTINGS
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/hubert-xlarge-ls960-ft"

cache_dir = Path("hubert_embeddings_incremental")
cache_dir.mkdir(exist_ok=True)
time_per_step = 0.02  # ~20 ms per frame

# settings for incremental extraction
BATCH_SIZE = 16            # how many truncated prefixes per forward pass (can be lowered)
                           
NORMALIZE_PER_PREFIX = False
# False (default): z-score the FULL waveform once, then truncate. Only the file's overall
#   mean/variance comes from later audio; no word-specific information does.
# True: z-score each truncated prefix on its own. Strictly no look-ahead, but
#   very short prefixes (the first few frames) get unstable scaling.

# ----------------------------
# LOAD MODEL + PROCESSOR
# ----------------------------
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = HubertModel.from_pretrained(model_name, output_hidden_states=True).to(device)
model.eval()

# Frame geometry of HuBERT's convolutional feature encoder.
# For HuBERT this is a 400-sample (25 ms) receptive field and 320-sample (20 ms) hop.
def _conv_geometry(model):
    rf, hop = 1, 1
    for k, s in zip(model.config.conv_kernel, model.config.conv_stride):
        rf += (k - 1) * hop
        hop *= s
    return rf, hop

RECEPTIVE_FIELD, HOP = _conv_geometry(model)


def _n_frames(n_samples):
    return int(model._get_feat_extract_output_lengths(torch.tensor(n_samples)).item())


# ----------------------------
# last-frame hidden states for a batch of prefixes
# ----------------------------
def _last_frame_hidden_states(prefixes):
    """
    prefixes: list of 1-D float tensors (already normalised), different lengths.
    Runs them as one zero-padded batch with an attention mask and returns a
    Tensor (n_prefixes, n_hidden_states, hidden_size) holding each prefix's
    LAST valid frame from every hidden state.

    Padding does not leak into valid frames: the conv encoder's layer norm acts
    per frame, the padded frames are zeroed before the positional convolution,
    and the attention mask hides them from self-attention. 
    """
    lens = [len(p) for p in prefixes]
    x = torch.zeros(len(prefixes), max(lens))
    mask = torch.zeros(len(prefixes), max(lens), dtype=torch.long)
    for i, p in enumerate(prefixes):
        x[i, : len(p)] = p
        mask[i, : len(p)] = 1

    with torch.no_grad():
        outputs = model(input_values=x.to(device), attention_mask=mask.to(device),
                        output_hidden_states=True)
    hs = torch.stack(outputs.hidden_states, dim=1)  # (B, n_hidden_states, T, D)

    last_idx = torch.tensor([_n_frames(n) - 1 for n in lens], device=hs.device)
    rows = torch.arange(len(prefixes), device=hs.device)
    return hs[rows, :, last_idx, :].cpu()  # (B, n_hidden_states, D)


# ----------------------------
# FUNCTION: extract + cache embeddings 
# ----------------------------
def extract_hubert_embeddings(audio_path: str, sr_target: int = 16000):
    """
    Decode audio via torchcodec, then run HuBERT incrementally and cache embeddings.
    Returns: dict[layer_idx] = Tensor of shape (T, hidden_size) 
    where row k comes from a forward pass on the audio up to the end of frame k.
    """
    out_path = cache_dir / f"{Path(audio_path).stem}_embeddings.pt"
    if out_path.exists():
        # load cached embeddings (on CPU)
        return torch.load(out_path, map_location="cpu")

    # Decode audio with torchcodec
    # Pass sample_rate argument so output is resampled to sr_target
    decoder = AudioDecoder(audio_path, sample_rate=sr_target)
    samples = decoder.get_all_samples()
    waveform = samples.data  # (num_channels, num_samples)
    sr = samples.sample_rate

    # If multi-channel, convert to mono by averaging
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0, keepdim=True)
    # shape becomes (1, num_samples)
    waveform = waveform.squeeze(0)  # shape: (num_samples,)

    # Preprocess with feature extractor  (keep the normalised waveform so it can be truncated)
    inputs = processor(waveform, sampling_rate=sr, return_tensors="pt")
    base = inputs["input_values"][0].float()  # (num_samples,), normalised over the full file

    # Forward passes on growing prefixes.
    #     The shortest prefix whose LAST frame is frame k has
    #     RECEPTIVE_FIELD + k * HOP samples. Frame count matches the full-file run.
    n_total = _n_frames(len(base))
    cut_points = [RECEPTIVE_FIELD + k * HOP for k in range(n_total)]


    last_frames = []
    for start in range(0, n_total, BATCH_SIZE):
        prefixes = []
        for n in cut_points[start:start + BATCH_SIZE]:
            if NORMALIZE_PER_PREFIX:
                p = processor(waveform[:n], sampling_rate=sr,
                              return_tensors="pt")["input_values"][0].float()
            else:
                p = base[:n]
            prefixes.append(p)
        last_frames.append(_last_frame_hidden_states(prefixes))
    last_frames = torch.cat(last_frames, dim=0)  # (T, n_hidden_states, D)

    # Build dict of embeddings
    embeddings = {i: last_frames[:, i, :].contiguous()
                  for i in range(last_frames.shape[1])}


    torch.save(embeddings, out_path)
    return embeddings

