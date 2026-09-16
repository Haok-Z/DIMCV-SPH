from pathlib import Path
import re
import subprocess
import shutil
from PIL import Image, ImageDraw
import imageio_ffmpeg

ROOT = Path(r"D:/Simulation/SourceCode/DIMCV-SPH")
RESULTS = ROOT / "results"
OUT = ROOT / "experiment_videos"
FPS = 30


def frame_key(p: Path):
    m = re.search(r"(\d+)(?=\.png$)", p.name)
    return int(m.group(1)) if m else p.name


def pngs(folder):
    return sorted(folder.glob("*.png"), key=frame_key)


def encode(frames, output, fps=FPS):
    if not frames:
        return False
    first = Image.open(frames[0]).convert("RGB")
    size = first.size
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
           str(output)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for path in frames:
            im = Image.open(path).convert("RGB")
            if im.size != size:
                im = im.resize(size, Image.Resampling.LANCZOS)
            proc.stdin.write(im.tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="replace")
        rc = proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed for {output}: {err[-1000:]}")
    return True


def encode_combined(fluid_frames, segment_frames, output, fps=FPS):
    if not fluid_frames or not segment_frames:
        return False
    fluid0 = Image.open(fluid_frames[0]).convert("RGB")
    seg0 = Image.open(segment_frames[0]).convert("RGB")
    h = max(fluid0.height, seg0.height)
    fw, sw = fluid0.width, seg0.width
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    size = (fw + sw, h)
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
           str(output)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        count = min(len(fluid_frames), len(segment_frames))
        for i in range(count):
            fi = Image.open(fluid_frames[i]).convert("RGB")
            si = Image.open(segment_frames[i]).convert("RGB")
            if fi.size != (fw, fluid0.height):
                fi = fi.resize((fw, fluid0.height), Image.Resampling.LANCZOS)
            if si.size != (sw, seg0.height):
                si = si.resize((sw, seg0.height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", size, "black")
            canvas.paste(fi, (0, 0))
            canvas.paste(si, (fw, 0))
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="replace")
        rc = proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed for {output}: {err[-1000:]}")
    return True


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    videos = []
    skipped = []
    for exp in sorted(RESULTS.iterdir()):
        if not exp.is_dir() or exp.name.startswith("."):
            continue
        method_dirs = [p for p in exp.iterdir() if p.is_dir() and (p / "images").is_dir()]
        for method in sorted(method_dirs):
            images_root = method / "images"
            regular = pngs(images_root)
            nested = {p.name: pngs(p) for p in images_root.iterdir() if p.is_dir()}
            exp_out = OUT / exp.name / method.name
            if regular:
                out = exp_out / f"{exp.name}_{method.name}_fluid.mp4"
                encode(regular, out)
                videos.append(out)
            for name, frames in sorted(nested.items()):
                if frames:
                    out = exp_out / f"{exp.name}_{method.name}_{name}.mp4"
                    encode(frames, out)
                    videos.append(out)
            seg = nested.get(next((n for n in nested if n.startswith("segments")), ""), [])
            fluid = nested.get(next((n for n in nested if n.startswith("sph")), ""), [])
            if seg and fluid:
                out = exp_out / f"{exp.name}_{method.name}_fluid_and_segments.mp4"
                encode_combined(fluid, seg, out)
                videos.append(out)
            if not regular and not nested:
                skipped.append(str(images_root))
    print(f"videos={len(videos)}")
    for p in videos:
        print(p)
    if skipped:
        print("skipped=")
        for p in skipped:
            print(p)

if __name__ == "__main__":
    main()
