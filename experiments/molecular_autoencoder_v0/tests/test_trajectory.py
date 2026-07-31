"""Frame sequences -> latent segments.

The invariant that matters is atom-composition consistency across frames. If
it silently breaks, residue r in frame 0 is not residue r in frame 1, the
latent sequence is misaligned, and nothing downstream can detect it -- the
diffusion model would simply learn noise. So it is checked, not assumed.
"""

import numpy as np
import pytest
import torch

from molae.dataset import sample_from_arrays
from molae.model import ModelConfig
from molae.model_direct import DirectAutoencoder
from molae.synthetic import make_synthetic_ala
from molae.trajectory import (
    InconsistentFrames, check_consistent, segments_from_frames, encode_segment,
    collate_segments, latent_drift, drift_ratio,
)


def _frames(n_frames=6, n_res=5, jitter=0.1, seed=0):
    """One molecule, many conformers: identical composition, moving coords."""
    rng = np.random.default_rng(seed)
    base = make_synthetic_ala(n_res=n_res)
    out = []
    for k in range(n_frames):
        f = {key: (np.array(v) if isinstance(v, np.ndarray) else v)
             for key, v in base.items()}
        # A smooth drift plus small noise, so consecutive frames are close.
        f["coords"] = (np.asarray(base["coords"], dtype=np.float32)
                       + k * jitter
                       + rng.normal(0, jitter * 0.05, np.asarray(base["coords"]).shape)
                       ).astype(np.float32)
        out.append(f)
    return out


def test_consistent_frames_pass():
    assert check_consistent(_frames()) == len(_frames()[0]["coords"])


def test_atom_count_change_is_caught():
    f = _frames()
    f[3]["coords"] = f[3]["coords"][:-1]
    with pytest.raises(InconsistentFrames, match="atoms"):
        check_consistent(f)


def test_composition_change_is_caught():
    """Same atom COUNT, different identity -- the silent case."""
    f = _frames()
    f[2]["residue_idx"] = np.asarray(f[2]["residue_idx"]).copy()
    f[2]["residue_idx"][0] = 7
    with pytest.raises(InconsistentFrames, match="residue_idx"):
        check_consistent(f)


def test_res_pos_reordering_is_caught():
    f = _frames()
    f[1]["res_pos"] = np.asarray(f[1]["res_pos"])[::-1].copy()
    with pytest.raises(InconsistentFrames, match="res_pos"):
        check_consistent(f)


def test_empty_frames_rejected():
    with pytest.raises(InconsistentFrames):
        check_consistent([])


def test_segments_are_non_overlapping_by_default():
    segs = segments_from_frames(list(range(10)), seg_len=4)
    assert segs == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_stride_gives_overlapping_segments():
    """Overlap multiplies training signal from a fixed trajectory, which
    matters because MD frames are expensive."""
    segs = segments_from_frames(list(range(8)), seg_len=4, stride=2)
    assert segs == [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]]


def test_too_few_frames_yields_no_segment():
    assert segments_from_frames(list(range(3)), seg_len=4) == []


def test_encode_segment_shape_and_frozen_codec():
    cfg = ModelConfig(d_model=32, n_heads=2, latent_dim=4, enc_self_layers=1,
                      dec_self_layers=1, encoder_type="direct")
    model = DirectAutoencoder(cfg)
    model.train()
    frames = _frames(n_frames=5, n_res=6)
    z = encode_segment(model, frames, torch.device("cpu"))
    assert z.shape[0] == 5 and z.shape[2] == 4
    # encode_segment must not leave the codec in eval mode behind the caller's
    # back, and must not require grad.
    assert model.training
    assert not z.requires_grad


def test_encode_segment_rejects_inconsistent_frames():
    cfg = ModelConfig(d_model=32, n_heads=2, latent_dim=4, enc_self_layers=1,
                      dec_self_layers=1, encoder_type="direct")
    frames = _frames()
    frames[1]["coords"] = frames[1]["coords"][:-2]
    with pytest.raises(InconsistentFrames):
        encode_segment(DirectAutoencoder(cfg), frames, torch.device("cpu"))


def test_collate_pads_and_masks():
    segs = [torch.randn(4, 6, 3), torch.randn(2, 5, 3)]
    b = collate_segments(segs)
    assert b["z"].shape == (2, 4, 6, 3)
    assert b["frame_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert b["group_mask"][1].tolist() == [1, 1, 1, 1, 1, 0]
    # padded region is zero, so a mask bug shows up as zeros not garbage
    assert torch.equal(b["z"][1, 2:], torch.zeros(2, 6, 3))


def test_drift_is_small_for_a_smooth_path():
    """The codec-side diagnostic: consecutive frames close relative to the
    segment's spread means the latent tracks conformational change."""
    t = torch.linspace(0, 1, 12)[:, None, None]
    smooth = t * torch.ones(1, 5, 4)
    assert drift_ratio(smooth) < 0.5


def test_drift_is_large_for_a_scrambled_path():
    torch.manual_seed(0)
    scrambled = torch.randn(12, 5, 4)
    assert drift_ratio(scrambled) > 1.0


def test_latent_drift_length_and_edge_cases():
    assert latent_drift(torch.randn(5, 3, 2)).shape == (4,)
    assert latent_drift(torch.randn(1, 3, 2)).numel() == 0
    assert np.isnan(drift_ratio(torch.randn(1, 3, 2)))
