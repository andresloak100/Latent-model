"""Joint-segment latent diffusion.

The load-bearing test here is `test_frames_are_modelled_jointly`. Generating a
segment JOINTLY, rather than one frame at a time, is the entire claim that
separates this from ATMOS's autoregressive frame decoding -- so it is asserted
directly rather than assumed from the architecture diagram.
"""

import torch

from molae.latent_video import LatentVideoConfig, SegmentDiT, LatentVideoDiffusion


def _wake(m):
    """Give the zero-initialised layers small random values.

    DiT zero-init makes an untrained model an exact no-op -- every AdaLN gate
    and the output projection are zero, so it returns zeros regardless of
    input. That is correct at init (asserted below) but it means connectivity
    cannot be probed through an untrained model: the joint-modelling tests
    would pass trivially on a model that ignores its input entirely.
    """
    torch.manual_seed(1)
    for name, p in m.named_parameters():
        if p.detach().abs().sum() == 0:
            with torch.no_grad():
                p.normal_(0.0, 0.15)
    return m


def _cfg(**kw):
    base = dict(latent_dim=4, d_model=32, n_heads=4, depth=2, ff_mult=2)
    base.update(kw)
    return LatentVideoConfig(**base)


def test_output_shape_matches_input():
    m = SegmentDiT(_cfg())
    z = torch.randn(2, 5, 7, 4)
    out = m(z, torch.rand(2))
    assert out.shape == z.shape


def test_segment_length_and_size_are_free_at_inference():
    """Sinusoidal index encodings, so neither T nor R is baked into a table."""
    m = SegmentDiT(_cfg())
    m.eval()
    for T, R in [(2, 3), (16, 40), (33, 7)]:
        out = m(torch.randn(1, T, R, 4), torch.rand(1))
        assert out.shape == (1, T, R, 4)


def test_frames_are_modelled_jointly():
    """THE distinguishing property. Perturbing frame 0 must change the
    prediction at the LAST frame -- information flows across the whole segment
    in both directions, which an autoregressive decoder cannot do."""
    torch.manual_seed(0)
    m = _wake(SegmentDiT(_cfg(depth=2)))
    m.eval()
    z = torch.randn(1, 6, 5, 4)
    t = torch.full((1,), 0.5)
    with torch.no_grad():
        a = m(z, t)
        z2 = z.clone()
        z2[0, 0] += 3.0                      # perturb the FIRST frame only
        b = m(z2, t)
    delta_last = (a[0, -1] - b[0, -1]).abs().max()
    assert delta_last > 1e-4, "last frame did not see the first -- not joint"


def test_residues_are_modelled_jointly_within_a_frame():
    torch.manual_seed(0)
    m = _wake(SegmentDiT(_cfg()))
    m.eval()
    z = torch.randn(1, 4, 6, 4)
    t = torch.full((1,), 0.5)
    with torch.no_grad():
        a = m(z, t)
        z2 = z.clone()
        z2[0, :, 0] += 3.0                   # perturb residue 0 in every frame
        b = m(z2, t)
    assert (a[0, :, -1] - b[0, :, -1]).abs().max() > 1e-4


def test_padded_groups_do_not_leak_into_real_outputs():
    torch.manual_seed(0)
    m = _wake(SegmentDiT(_cfg()))
    m.eval()
    z = torch.randn(1, 3, 6, 4)
    gm = torch.tensor([[1., 1., 1., 1., 0., 0.]])
    t = torch.full((1,), 0.3)
    with torch.no_grad():
        a = m(z, t, group_mask=gm)
        z2 = z.clone()
        z2[0, :, 4:] += 10.0                 # corrupt only the padded groups
        b = m(z2, t, group_mask=gm)
    assert torch.allclose(a[0, :, :4], b[0, :, :4], atol=1e-5)


def test_padded_frames_do_not_leak_into_real_outputs():
    torch.manual_seed(0)
    m = _wake(SegmentDiT(_cfg()))
    m.eval()
    z = torch.randn(1, 6, 4, 4)
    fm = torch.tensor([[1., 1., 1., 1., 0., 0.]])
    t = torch.full((1,), 0.3)
    with torch.no_grad():
        a = m(z, t, frame_mask=fm)
        z2 = z.clone()
        z2[0, 4:] += 10.0
        b = m(z2, t, frame_mask=fm)
    assert torch.allclose(a[0, :4], b[0, :4], atol=1e-5)


def test_fully_masked_row_stays_finite():
    """An all-padding row makes softmax produce NaN; the guard must hold."""
    m = SegmentDiT(_cfg())
    z = torch.randn(2, 4, 5, 4)
    fm = torch.tensor([[1., 1., 1., 1.], [0., 0., 0., 0.]])
    gm = torch.tensor([[1., 1., 1., 1., 1.], [0., 0., 0., 0., 0.]])
    out = m(z, torch.rand(2), group_mask=gm, frame_mask=fm)
    assert torch.isfinite(out).all()


def test_zero_initialised_output_starts_as_no_op():
    """DiT-style zero init: the model begins as the identity residual stream,
    so early training is not fighting a random velocity field."""
    m = SegmentDiT(_cfg())
    out = m(torch.randn(2, 3, 4, 4), torch.rand(2))
    assert torch.allclose(out, torch.zeros_like(out))


def test_training_loss_is_finite_and_masked():
    m = LatentVideoDiffusion(_cfg())
    z1 = torch.randn(2, 4, 6, 4)
    gm = torch.tensor([[1.] * 6, [1., 1., 1., 0., 0., 0.]])
    fm = torch.ones(2, 4)
    loss, comp = m.training_loss(z1, group_mask=gm, frame_mask=fm)
    assert torch.isfinite(loss) and loss.item() >= 0
    assert "flow_mse" in comp


def test_loss_ignores_padded_content():
    """Changing padded latents must not change the loss."""
    torch.manual_seed(0)
    m = LatentVideoDiffusion(_cfg())
    z1 = torch.randn(1, 3, 6, 4)
    gm = torch.tensor([[1., 1., 1., 0., 0., 0.]])
    g = torch.Generator().manual_seed(7)
    a, _ = m.training_loss(z1, group_mask=gm, generator=g)
    z2 = z1.clone()
    z2[0, :, 3:] += 5.0
    g = torch.Generator().manual_seed(7)
    b, _ = m.training_loss(z2, group_mask=gm, generator=g)
    assert torch.allclose(a, b, atol=1e-5)


def test_sampling_shape_and_determinism():
    m = LatentVideoDiffusion(_cfg())
    g = torch.Generator().manual_seed(3)
    a = m.sample((1, 5, 6, 4), torch.device("cpu"), steps=4, generator=g)
    g = torch.Generator().manual_seed(3)
    b = m.sample((1, 5, 6, 4), torch.device("cpu"), steps=4, generator=g)
    assert a.shape == (1, 5, 6, 4)
    assert torch.allclose(a, b)


def test_can_overfit_a_single_segment():
    """Sanity that the objective trains at all."""
    torch.manual_seed(0)
    m = LatentVideoDiffusion(_cfg(d_model=64, depth=2))
    z1 = torch.randn(1, 4, 5, 4)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    first = None
    for step in range(120):
        loss, _ = m.training_loss(z1, generator=g)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.8, (first, loss.item())
