"""Chunks (LIIF) and tiles (CLIF) only bound memory: the rendered volume does not depend on them."""
import torch
from c3d.models.heads.liif_3d import LIIFHead3D, LIIFHeadConfig
from c3d.models.heads.clif_3d import CLIFHead3D, CLIFHeadConfig


def test_liif_chunk_invariance():
    torch.manual_seed(0)
    head = LIIFHead3D(6, LIIFHeadConfig(mlp_hidden=(16, 16))).eval()
    feat = torch.randn(1, 6, 4, 5, 6)
    with torch.no_grad():
        a = head.render(feat, (16, 20, 24), chunk=10_000_000)
        b = head.render(feat, (16, 20, 24), chunk=777)
    assert a.shape == (1, 1, 16, 20, 24)
    assert torch.allclose(a, b, atol=1e-6)


def test_clif_tiles_match_full_grid():
    torch.manual_seed(0)
    head = CLIFHead3D(6, CLIFHeadConfig(dim=3, hidden_channels=8, num_res_blocks=2, norm="channel")).eval()
    feat = torch.randn(1, 6, 4, 4, 4)
    with torch.no_grad():
        full = head.render(feat, (16, 16, 16), tile_size=(16, 16, 16), halo=6)
        tiled = head.render(feat, (16, 16, 16), tile_size=(5, 7, 6), halo=6)
    assert torch.allclose(full, tiled, atol=1e-5)


def test_liif_any_resolution():
    head = LIIFHead3D(6, LIIFHeadConfig(mlp_hidden=(8,))).eval()
    feat = torch.randn(1, 6, 4, 4, 4)
    with torch.no_grad():
        for shape in ((8, 8, 8), (13, 17, 9), (32, 32, 32)):
            assert head.render(feat, shape).shape[2:] == shape
