"""Verify the selected checkout, CUDA binary ABI and local model assets."""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    os.chdir(ROOT)
    import importlib.metadata as md
    import torch
    import hmr4d
    from pytorch3d.ops import knn_points
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
    from hmr4d.utils.runtime_paths import ffmpeg_program

    print("Python:", sys.executable)
    print("Source:", hmr4d.__file__)
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("The local PyTorch3D wheel needs Python 3.10")
    if torch.__version__ != "2.11.0+cu130":
        raise RuntimeError(f"Expected torch 2.11.0+cu130, got {torch.__version__}")
    for name in ("torch", "torchvision", "pytorch3d", "numpy", "chumpy", "ultralytics"):
        print(name, md.version(name))
    print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_arch_list())
    points = torch.tensor([[[0., 0., 1.], [1., 0., 1.]]], device="cuda")
    distances = knn_points(points, points).dists
    assert torch.allclose(distances, torch.zeros_like(distances))
    vertices = torch.tensor([[-.8, -.8, 1.], [.8, -.8, 1.], [0., .8, 1.]], device="cuda")
    faces = torch.tensor([[0, 1, 2]], device="cuda")
    fragments = rasterize_meshes(Meshes(verts=[vertices], faces=[faces]), image_size=32)
    assert bool((fragments[0] >= 0).any())
    torch.cuda.synchronize()
    for name in ("ffmpeg", "ffprobe"):
        print(name, ffmpeg_program(name))
    assets = ["gvhmr/gvhmr_siga24_release.ckpt", "hmr2/epoch=10-step=25000.ckpt",
              "vitpose/vitpose-h-multi-coco.pth", "yolo/yolov8x.pt",
              "body_models/smplx/SMPLX_NEUTRAL.npz", "body_models/smpl/SMPL_NEUTRAL.pkl"]
    for name in assets:
        if not (ROOT / "inputs/checkpoints" / name).is_file():
            raise FileNotFoundError(f"Missing asset: {name}; run tools/windows/prepare_assets.py")
    from hmr4d.utils.smplx_utils import make_smplx
    assert make_smplx("smpl").bm.gender == "neutral"
    assert make_smplx("supermotion").bm.gender == "neutral"
    print("PASS: neutral models, CUDA KNN/rasterizer, local assets and video tools")


if __name__ == "__main__":
    main()
