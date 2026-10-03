"""Procedural video rendering engine built on ffmpeg.

Implements the skill's built-in `text_to_video` and `image_to_video`
operations locally so the application works with zero external API keys.
Every render produces a real, playable H.264/AAC MP4 in the requested
aspect ratio and duration, with BBU CHANNEL branding baked in as an overlay.

When `MODELARK_API_KEY` is configured, `ai_provider` can route the same
request to the real Seedance 2.5 model instead (see app/ai_provider.py).
"""
from __future__ import annotations

import hashlib
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import config, prompt_utils

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

_PALETTES: list[tuple[str, list[str]]] = [
    ("night", ["0x0b1026", "0x1b2a4a", "0x3b1e5e"]),
    ("sunset", ["0xff7e5f", "0xfeb47b", "0x2b1055"]),
    ("ocean", ["0x02111b", "0x0a4d68", "0x088395"]),
    ("forest", ["0x0b1d13", "0x1e4620", "0x7cb518"]),
    ("neon", ["0x1b2a4a", "0x8a2be2", "0xff6ec7"]),
    ("warm", ["0x2b1055", "0xff6b35", "0xffd166"]),
]


class RenderError(RuntimeError):
    """Raised when ffmpeg fails to produce a video."""


@dataclass
class RenderResult:
    path: Path
    filename: str
    duration: int
    width: int
    height: int
    ratio: str
    engine: str
    prompt: str
    model_version: str
    source_image: str | None = None


def _seed_for(prompt: str) -> int:
    return int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)


def _pick_palette(prompt: str, analysis: dict) -> list[str]:
    if analysis.get("is_night"):
        name = "night"
    elif analysis.get("is_sunset"):
        name = "sunset"
    elif analysis.get("is_ocean"):
        name = "ocean"
    elif analysis.get("is_forest"):
        name = "forest"
    else:
        name = _PALETTES[_seed_for(prompt) % len(_PALETTES)][0]
    for pname, colors in _PALETTES:
        if pname == name:
            return colors
    return _PALETTES[-1][1]


def _title_for(prompt: str, max_words: int = 5) -> str:
    words = prompt_utils.slugify(prompt, max_len=60).replace("_", " ").split()
    title = " ".join(words[:max_words]).upper()
    return title or "AI VIDEO"


def _dimensions(ratio: str) -> tuple[int, int]:
    return config.RATIO_DIMENSIONS.get(ratio, config.RATIO_DIMENSIONS[config.DEFAULT_RATIO])


def grade_filters() -> list[str]:
    """Photographic grade shared by every engine that animates AI stills.

    Massively upscaled AI images look flat and synthetic on their own. A film
    curve, local contrast, subtle temporal blend, grain and a soft vignette are
    what make the result read as camera footage rather than a moving picture.
    """
    if not config.REALISM:
        return ["eq=saturation=1.12:contrast=1.06"]
    grade = [
        "curves=preset=medium_contrast",
        "eq=contrast=1.06:saturation=1.12:brightness=0.005:gamma=0.99",
        f"unsharp=5:5:{config.REALISM_SHARPEN}:5:5:0.0",
    ]
    if config.REALISM_TEMPORAL > 0:
        grade.append(
            f"tmix=frames=2:weights='{1 - config.REALISM_TEMPORAL} {config.REALISM_TEMPORAL}'"
        )
    if config.REALISM_GRAIN > 0:
        grade.append(f"noise=alls={config.REALISM_GRAIN}:allf=t+u")
    if config.REALISM_VIGNETTE > 0:
        grade.append(f"vignette=angle=PI*{config.REALISM_VIGNETTE / 2:.4f}")
    return grade


def _write_textfile(tmpdir: Path, name: str, text: str) -> Path:
    p = tmpdir / name
    p.write_text(text, encoding="utf-8")
    return p


def _drawtext(textfile: Path, fontfile: str, size_expr: str, y_expr: str,
              alpha: str = "1.0", enable: str | None = None,
              box: bool = False) -> str:
    parts = [
        f"textfile={textfile}",
        f"fontfile={fontfile}",
        f"fontsize={size_expr}",
        "fontcolor=white@{a}".format(a=alpha),
        "x=(w-text_w)/2",
        f"y={y_expr}",
    ]
    if box:
        parts.append("box=1")
        parts.append("boxcolor=black@0.35")
        parts.append("boxborderw=18")
    if enable:
        parts.append(f"enable='{enable}'")
    return "drawtext=" + ":".join(parts)


def build_command(
    *,
    prompt: str,
    duration: int,
    ratio: str,
    output: Path,
    text_dir: Path,
    image_path: Path | None = None,
) -> list[str]:
    width, height = _dimensions(ratio)
    analysis = prompt_utils.motion_analysis(prompt)
    colors = _pick_palette(prompt, analysis)
    speed = 0.03 if analysis.get("has_motion") else 0.012

    title = _title_for(prompt)
    subtitle = f"AI Generated • {duration}s • {ratio} • Seedance 2.5"
    watermark = config.BBU["watermark_overlay_template"]

    title_file = _write_textfile(text_dir, "title.txt", title)
    sub_file = _write_textfile(text_dir, "subtitle.txt", subtitle)
    mark_file = _write_textfile(text_dir, "watermark.txt", watermark)

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats"]

    if image_path is not None:
        cmd += ["-loop", "1", "-t", str(duration), "-i", str(image_path)]
        base = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase"
            f":flags={config.REALISM_UPSCALE},"
            f"crop={width}:{height},setsar=1"
        )
        motion = (
            f"zoompan=z='min(zoom+0.0006,1.20)'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d=1:s={width}x{height}:fps={config.FPS}"
        )
    else:
        cmd += [
            "-f", "lavfi", "-i",
            (
                f"gradients=s={width}x{height}:c0={colors[0]}:c1={colors[1]}"
                f":c2={colors[2]}:n=3:type=spiral:speed={speed}"
                f":seed={_seed_for(prompt)}:duration={duration}:rate={config.FPS}"
            ),
        ]
        base = (
            f"noise=alls=10:allf=t+u,"
            f"vignette=PI/5,format=yuv420p"
        )
        motion = (
            f"zoompan=z='min(zoom+0.0005,1.18)'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d=1:s={width}x{height}:fps={config.FPS}"
        )

    # Ambient audio bed so the MP4 is a complete, shareable clip.
    audio_idx = 1 if image_path is not None else 1
    cmd += [
        "-f", "lavfi", "-i",
        f"anoisesrc=d={duration}:c=pink:r=48000:a=0.015",
    ]

    # The grade only makes sense for real stills; the procedural gradient
    # background has its own look.
    graded = list(grade_filters()) if image_path is not None else []
    video_filters = ",".join([
        base,
        motion,
        *graded,
        "format=yuv420p",
        _drawtext(title_file, FONT_BOLD, f"h/12", "h*0.38",
                  box=True, enable="lt(t,4)"),
        _drawtext(sub_file, FONT_REGULAR, f"h/30", "h*0.52",
                  alpha="0.9", enable="lt(t,4)"),
        _drawtext(mark_file, FONT_BOLD, f"h/34", "h-text_h-28",
                  alpha="0.75", box=True),
    ])

    cmd += [
        "-filter_complex", f"[0:v]{video_filters}[v]",
        "-map", "[v]",
        "-map", f"{audio_idx}:a",
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "96k",
        "-shortest",
        str(output),
    ]
    return cmd


def render(
    *,
    prompt: str,
    duration: int,
    ratio: str,
    filename: str,
    image_path: Path | None = None,
) -> RenderResult:
    """Render one video synchronously and return the result."""
    output = config.OUTPUT_DIR / filename
    text_dir = Path(tempfile.mkdtemp(prefix="aivm_"))
    try:
        cmd = build_command(
            prompt=prompt,
            duration=duration,
            ratio=ratio,
            output=output,
            text_dir=text_dir,
            image_path=image_path,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not output.exists():
            raise RenderError(
                f"ffmpeg failed (rc={proc.returncode}): {proc.stderr.strip()[:800]}"
            )
    finally:
        for f in text_dir.glob("*"):
            f.unlink(missing_ok=True)
        text_dir.rmdir()

    width, height = _dimensions(ratio)
    return RenderResult(
        path=output,
        filename=filename,
        duration=duration,
        width=width,
        height=height,
        ratio=ratio,
        engine="procedural",
        prompt=prompt,
        model_version=config.FORCED_MODEL_VERSION,
        source_image=str(image_path) if image_path else None,
    )


def probe(path: Path) -> dict:
    """Return real media metadata for a rendered file."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration,size",
            "-show_entries", "stream=codec_name,width,height",
            "-of", "default=noprint_wrappers=1",
            str(path),
        ],
        capture_output=True, text=True,
    )
    info: dict = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k] = v
    return info
