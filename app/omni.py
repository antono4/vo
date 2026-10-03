"""Multi-shot "Omni-lite" rendering engine.

The `classic` engine zooms a single still for the whole clip, which does not
read as AI video. This engine instead builds a real sequence: one AI image per
shot, a different camera move per shot (push-in, pan, crane, pull-back, drift,
orbit), a crossfade between shots, a synthesized ambient bed with a downbeat on
every cut, and a title card that fades out.

It deliberately avoids `zoompan`: that filter runs at the input frame rate and
is very slow on multi-second clips. Per-shot motion is expressed with
`scale` + `crop` driven by `t`/`n`, which renders in seconds rather than
minutes and gives the same push/pan/crane look.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import config, renderer

FONT_BOLD = renderer.FONT_BOLD
FONT_REGULAR = renderer.FONT_REGULAR


class OmniRenderError(RuntimeError):
    """Raised when the omni pipeline cannot produce a video."""


@dataclass
class OmniSceneInput:
    image: Path
    camera: str
    index: int


@dataclass
class OmniRenderResult:
    path: Path
    filename: str
    duration: int
    width: int
    height: int
    ratio: str
    engine: str
    prompt: str
    scene_count: int
    scene_prompts: list[str]
    plan_source: str


def _even(n: float) -> int:
    v = int(round(n))
    return v - (v % 2)


def _smoothstep(T: str, span: float) -> str:
    """Eased 0..1 ramp over `span` seconds starting at 0.

    A linear ramp reads as a mechanical zoom; easing in and out is what makes a
    move feel like a camera operator starting and stopping. `T` is the clip time
    expression (frame counter / fps).
    """
    p = f"min({T}/{span:.4f},1)"
    return f"({p}*{p}*(3-2*{p}))"


def camera_filter(camera: str, width: int, height: int, fps: int) -> str:
    """Per-shot motion + photographic grade.

    `zoompan` does not expose a `t` variable, so clip time is derived from the
    output frame counter: `T = on / fps`. Moves are eased (see `_smoothstep`) and
    travel ~10% so they read as deliberate but never expose the frame edges; the
    input is upscaled first so the moving crop always has coverage.

    The grade chain afterwards is what sells the "real footage" look: a
    filmic curve, local contrast (unsharp), grain and a soft vignette. Without
    it the massively upscaled AI stills look obviously synthetic.
    """
    cam = (camera or "").lower()
    T = f"on/{fps}"
    span = 4.0
    e = _smoothstep(T, span)
    z = f"1+0.10*{e}"
    if "pan" in cam or "orbit" in cam:
        x = f"(iw-iw/zoom)*(0.5+0.5*{e})"
        y = "ih/2-(ih/zoom/2)"
    elif "crane" in cam or "rise" in cam:
        x = "iw/2-(iw/zoom/2)"
        y = f"(ih-ih/zoom)*(1-{e})"
    elif "pull" in cam or "back" in cam or "reveal" in cam:
        z = f"1.10-0.10*{e}"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif "drift" in cam or "handheld" in cam:
        z = "1.06"
        x = f"(iw-iw/zoom)*(0.5+0.03*sin({T}*1.5))"
        y = f"(ih-ih/zoom)*(0.5+0.03*cos({T}*1.3))"
    else:  # push-in (default)
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"

    chain = [
        f"scale={_even(width * 2)}:{_even(height * 2)}"
        f":force_original_aspect_ratio=increase:flags={config.REALISM_UPSCALE}",
        f"crop={width}:{height}",
        "setsar=1",
        f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={width}x{height}:fps={fps}",
    ]
    chain.extend(_grade_filters())
    chain.append("format=yuv420p")
    chain.append("setsar=1")
    return ",".join(chain)


def _grade_filters() -> list[str]:
    """Photographic grade applied to every shot (shared with the classic engine)."""
    return renderer.grade_filters()


def _xfade_transition(i: int) -> str:
    """Pick a dissolve-family transition for join `i`.

    Hard wipes look like a slideshow, so joins alternate between several soft
    dissolve/wipe flavours to keep long sequences from feeling repetitive.
    """
    options = ["fade", "dissolve", "smoothleft", "fadeblack", "smoothright"]
    return options[i % len(options)]


def segment_geometry(duration: int, n: int) -> tuple[float, float, float]:
    """Return (segment_seconds, xfade_seconds, cut_offset_seconds).

    `n` shots of length `seg`, each overlapped by `xfade`, sum to `duration`:
    n*seg - (n-1)*xfade == duration. Cut i happens at i*(seg - xfade).
    """
    total = float(duration)
    if n <= 1:
        return total, 0.0, total
    xfade = min(config.OMNI_XFADE_SECONDS, total / n * 0.5)
    seg = (total + xfade * (n - 1)) / n
    return seg, xfade, seg - xfade


def ambient_audio_filter(duration: int, cut_offset: float) -> str:
    """Low ambient bed plus a downbeat on each cut, faded in and out.

    Each beat term is gated with `(t>b)` so it contributes nothing before its
    cut — without the gate `exp(-k*(t-b))` blows up for t < b.
    """
    beats: list[float] = []
    t = cut_offset
    while t < duration - 0.5 and len(beats) < 64:
        beats.append(t)
        t += cut_offset
    if beats:
        tone = "+".join(
            f"gt(t,{b:.2f})*0.14*exp(-2.5*(t-{b:.2f}))*sin(2*PI*70*(t-{b:.2f}))"
            for b in beats
        )
    else:
        tone = "0"
    fade_out = max(0.0, duration - 1.5)
    return (
        f"sine=frequency=110:sample_rate=48000:duration={duration}[a1];"
        f"anoisesrc=d={duration}:c=brown:r=48000:a=0.035[a2];"
        f"[a1][a2]amix=inputs=2:duration=first:weights=0.35 1[ab];"
        f"[ab]volume='min(1.0,0.55+{tone})':eval=frame,"
        f"afade=t=in:st=0:d=1.2,afade=t=out:st={fade_out:.2f}:d=1.5"
    )


def _title_drawtext(textfile: Path, fontfile: str, size_expr: str, y_expr: str,
                    alpha_expr: str, enable: str, box: bool = False) -> str:
    parts = [
        f"textfile={textfile}",
        f"fontfile={fontfile}",
        f"fontsize={size_expr}",
        "fontcolor=white",
        f"alpha='{alpha_expr}'",
        "x=(w-text_w)/2",
        f"y={y_expr}",
        f"enable='{enable}'",
    ]
    if box:
        parts += ["box=1", "boxcolor=black@0.35", "boxborderw=18"]
    return "drawtext=" + ":".join(parts)


def build_command(
    *,
    scene_inputs: list[OmniSceneInput],
    duration: int,
    ratio: str,
    output: Path,
    text_dir: Path,
    title: str,
    subtitle: str,
    watermark: str,
) -> list[str]:
    if not scene_inputs:
        raise OmniRenderError("no scenes to render")

    width, height = renderer._dimensions(ratio)
    fps = config.FPS
    n = len(scene_inputs)
    seg, xfade, cut = segment_geometry(duration, n)

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats"]
    for si in scene_inputs:
        cmd += ["-loop", "1", "-t", f"{seg:.3f}", "-i", str(si.image)]

    title_file = renderer._write_textfile(text_dir, "omni_title.txt", title)
    sub_file = renderer._write_textfile(text_dir, "omni_sub.txt", subtitle)
    mark_file = renderer._write_textfile(text_dir, "omni_mark.txt", watermark)

    parts: list[str] = []
    for i, si in enumerate(scene_inputs):
        parts.append(f"[{i}:v]{camera_filter(si.camera, width, height, fps)}[v{i}]")

    if n == 1:
        last = "v0"
    else:
        prev = "v0"
        for i in range(1, n):
            out = f"x{i}"
            parts.append(
                f"[{prev}][v{i}]xfade=transition={_xfade_transition(i)}"
                f":duration={xfade:.3f}:offset={cut * i:.3f}[{out}]"
            )
            prev = out
        last = prev

    ts = config.OMNI_TITLE_SECONDS
    enable = f"between(t,0,{ts})"
    title_alpha = f"if(lt(t,{ts - 0.8:.2f}),1,max(0,({ts:.2f}-t)/0.8))"
    parts.append(
        f"[{last}]{_title_drawtext(title_file, FONT_BOLD, 'h/12', 'h*0.38', title_alpha, enable, box=True)}[t1]"
    )
    parts.append(
        f"[t1]{_title_drawtext(sub_file, FONT_REGULAR, 'h/30', 'h*0.52', title_alpha, enable)}[t2]"
    )
    parts.append(
        f"[t2]{renderer._drawtext(mark_file, FONT_BOLD, 'h/34', 'h-text_h-28', alpha='0.75', box=True)}[vout]"
    )
    parts.append(f"{ambient_audio_filter(duration, cut)}[aout]")

    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", "[vout]",
        "-map", "[aout]",
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "19",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-metadata", f"title={title}",
        "-metadata", f"comment=AI Video Maker (Omni-lite) | {subtitle}",
        str(output),
    ]
    return cmd


def render(
    *,
    prompt: str,
    duration: int,
    ratio: str,
    filename: str,
    scene_images: list[tuple[Path, str]],
    scene_prompts: list[str] | None = None,
    plan_source: str = "heuristic",
) -> OmniRenderResult:
    """Render the multi-shot video synchronously."""
    if not scene_images:
        raise OmniRenderError("no scene images supplied")

    output = config.OUTPUT_DIR / filename
    text_dir = Path(tempfile.mkdtemp(prefix="aivm_omni_"))
    scene_inputs = [
        OmniSceneInput(image=img, camera=cam, index=i)
        for i, (img, cam) in enumerate(scene_images)
    ]
    width, height = renderer._dimensions(ratio)
    title = renderer._title_for(prompt)
    subtitle = f"AI Generated • {duration}s • {ratio} • {len(scene_images)} shots • Omni"

    try:
        cmd = build_command(
            scene_inputs=scene_inputs,
            duration=duration,
            ratio=ratio,
            output=output,
            text_dir=text_dir,
            title=title,
            subtitle=subtitle,
            watermark=config.BBU["watermark_overlay_template"],
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not output.exists():
            raise OmniRenderError(
                f"ffmpeg failed (rc={proc.returncode}): {proc.stderr.strip()[:800]}"
            )
    finally:
        for f in text_dir.glob("*"):
            f.unlink(missing_ok=True)
        text_dir.rmdir()

    return OmniRenderResult(
        path=output,
        filename=filename,
        duration=duration,
        width=width,
        height=height,
        ratio=ratio,
        engine="omni",
        prompt=prompt,
        scene_count=len(scene_images),
        scene_prompts=list(scene_prompts or []),
        plan_source=plan_source,
    )
