"""One-command BVH refinement, previews, mesh audit and Blender/FBX export."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hmr4d.utils.mushroom_config import apply_constraints, read_constraints

DEFAULT_BVH = r"D:\track_dataset\Flare_bvh\marker53_optimized_ground.bvh"
DEFAULT_OUTPUT = r"D:\track_dataset\GVHMR_correct_results"
DEFAULT_BLENDER = r"D:\Blender Foundation\Blender 5.1\blender.exe"
DEFAULT_ADDON = r"D:\Blender Foundation\smplx_blender_addon-1.0.3-20260511\smplx_blender_addon"


def annotate(frame):
    """Return original-resolution landmarks and a user-selected background ROI."""
    import cv2

    scale = min(1.0, 1200 / frame.shape[1], 800 / frame.shape[0])
    display = cv2.resize(frame, None, fx=scale, fy=scale)
    name = "Static background: drag a rectangle, ENTER to accept, C to cancel"
    x, y, w, h = cv2.selectROI(name, display, showCrosshair=True)
    cv2.destroyAllWindows()
    if w <= 0 or h <= 0:
        raise ValueError("Background annotation cancelled")
    roi = [round(v / scale) for v in (x, y, x + w, y + h)]
    points = []
    labels = ["Top axis center", "Left cap rim", "Right cap rim", "Base front ground point"]
    name = "Mushroom: click 4 points; R resets; ENTER accepts; ESC cancels"
    cv2.namedWindow(name)

    def click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([round(x / scale), round(y / scale)])

    cv2.setMouseCallback(name, click)
    try:
        while True:
            view = display.copy()
            for i, (x, y) in enumerate(points):
                pos = (round(x * scale), round(y * scale))
                cv2.circle(view, pos, 5, (0, 255, 0), -1)
                cv2.putText(view, str(i + 1), pos, 0, 0.8, (0, 255, 0), 2)
            title = labels[len(points)] if len(points) < 4 else "ENTER to accept"
            cv2.putText(view, title, (10, 30), 0, 0.8, (0, 255, 255), 2)
            cv2.imshow(name, view)
            key = cv2.waitKey(30) & 255
            if key == 27 or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError("Mushroom annotation cancelled")
            if key in (ord("r"), ord("R")):
                points.clear()
            if key in (10, 13) and len(points) == 4:
                return points, roi
    finally:
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Existing GVHMR result folder")
    for name in ("start", "end", "circle-start", "circle-end"):
        p.add_argument("--" + name, type=int, required=True, help="Original zero-based frame; end is exclusive")
    p.add_argument("--output-root", default=DEFAULT_OUTPUT)
    p.add_argument("--bvh", default=DEFAULT_BVH)
    p.add_argument(
        "--bvh-range",
        nargs=2,
        type=int,
        default=[740, 1170],
        help="Stable circles in the BVH; change when using another BVH",
    )
    p.add_argument("--config", help="Optional per-video annotations JSON; command line frame ranges take precedence")
    p.add_argument("--constraints", help="Reusable constraint JSON from configure_mushroom_constraints.py")
    p.add_argument("--reference-frame", type=int)
    p.add_argument("--camera", choices=["jitter", "static"], default="jitter")
    p.add_argument("--iterations", type=int, help="Override base.iterations in the constraint config")
    p.add_argument("--blender", default=DEFAULT_BLENDER)
    p.add_argument("--addon", default=DEFAULT_ADDON)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--full-diagnostics",
        action="store_true",
        help="Retain large per-frame surface caches; compact diagnostics are the default",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Validate and print commands without writing or running optimization"
    )
    args = p.parse_args()
    folder = Path(args.input).resolve()
    if folder.is_file():
        folder = folder.parent
    output_root = Path(args.output_root).resolve()
    output = output_root / folder.name
    if output == folder or folder in output.parents:
        p.error("Use a separate output root to preserve source data")
    for path in [
        folder / "hmr4d_results.pt",
        folder / "preprocess/vitpose.pt",
        folder / "preprocess/bbx.pt",
        folder / "0_input_video.mp4",
        Path(args.bvh),
        Path(args.blender),
        Path(args.addon) / "__init__.py",
    ]:
        if not path.is_file():
            p.error(f"Missing dependency: {path}")
    if output.exists() and any(output.iterdir()) and not args.overwrite and not args.dry_run:
        p.error(f"Output exists: {output}. Use another --output-root or explicitly --overwrite.")
    import cv2

    cap = cv2.VideoCapture(str(folder / "0_input_video.mp4"))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (0 <= args.start <= args.circle_start < args.circle_end <= args.end <= count):
        cap.release()
        p.error(f"Require 0 <= start <= circle-start < circle-end <= end <= {count}")
    if fps <= 0 or (args.iterations is not None and args.iterations <= 0):
        p.error("Invalid video fps or iteration count")
    setup = output_root / "_setup" / folder.name
    config_path = setup / "annotations.json"
    cfg = {}
    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    elif config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if cfg.get("source_folder") != str(folder):
            p.error("Cached annotations belong to another input. Specify --config or another output root.")
    try:
        cfg = apply_constraints(cfg, read_constraints(args.constraints) if args.constraints else None)
        if args.iterations is not None:
            cfg["constraints"]["base"]["iterations"] = args.iterations
    except (ValueError, OSError) as exc:
        cap.release()
        p.error(str(exc))
    reference = args.reference_frame if args.reference_frame is not None else cfg.get("reference_frame", args.start)
    if not 0 <= reference < count:
        p.error("Reference frame is outside the video")
    if cfg.get("reference_frame", reference) != reference:
        cfg.pop("apparatus_pixels", None)
        cfg.pop("camera_roi", None)
    cap.set(cv2.CAP_PROP_POS_FRAMES, reference)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        p.error("Cannot decode annotation frame")
    if not cfg.get("apparatus_pixels") or not cfg.get("camera_roi"):
        if args.dry_run:
            p.error(
                "New video needs annotations. Run normally for interactive setup, or provide --config with apparatus_pixels and camera_roi."
            )
        print("Select stable background, then top center / left rim / right rim / base front ground point.", flush=True)
        cfg["apparatus_pixels"], cfg["camera_roi"] = annotate(frame)
    if cfg.get("keep_range") != [args.start, args.end] or cfg.get("circle_range") != [
        args.circle_start,
        args.circle_end,
    ]:
        # Do not silently transfer sample contact intervals when stage boundaries change.
        for key in ("evaluation_cycle_boundaries", "extra_hand_contacts", "ground_contacts", "occluded_keypoints"):
            cfg.pop(key, None)
    cfg.update(
        keep_range=[args.start, args.end],
        circle_range=[args.circle_start, args.circle_end],
        fps=fps,
        image_size=[frame.shape[1], frame.shape[0]],
        reference_frame=reference,
        bvh_circle_range=args.bvh_range,
        source_folder=str(folder),
    )
    cfg.setdefault("preparation_only", False)
    cfg["full_diagnostics"] = args.full_diagnostics
    cfg["periodic_weight"] = 0.0  # Experimental trajectory fitting has been removed.
    camera = setup / "camera"

    def run(script, *options):
        command = [sys.executable, str(ROOT / "tools" / script), *map(str, options)]
        execute(command)

    def execute(command):
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

    if not args.dry_run:
        setup.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    camera_args = []
    if args.camera == "jitter":
        run(
            "estimate_mushroom_camera.py",
            "--input",
            folder,
            "--output",
            camera,
            "--roi",
            *cfg["camera_roi"],
            "--reference-frame",
            reference,
        )
        camera_args = ["--camera-motion", camera / "camera_motion.npz"]
    run(
        "refine_mushroom.py",
        "--input",
        folder,
        "--bvh",
        args.bvh,
        "--config",
        config_path,
        "--output-root",
        output_root,
        "--iterations",
        cfg["constraints"]["base"]["iterations"],
        *camera_args,
        *(["--overwrite"] if args.overwrite else []),
    )
    for stage in ("legs", "feet"):
        if cfg["constraints"][stage]["enabled"]:
            run("refine_mushroom.py", "--stage", stage, "--input", output, "--bvh", args.bvh, "--overwrite")
    run("preview_mushroom.py", "--input", output)
    run("audit_mushroom_mesh.py", "--input", output)
    execute(
        [
            args.blender,
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(ROOT / "tools/export_mushroom_blender.py"),
            "--",
            "--input",
            str(output),
            "--addon",
            args.addon,
        ]
    )
    print(("DRY RUN: " if args.dry_run else "Completed: ") + str(output))


if __name__ == "__main__":
    main()
