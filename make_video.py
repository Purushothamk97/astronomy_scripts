"""
Astronomy Video Generator — YouTube-Ready (1920x1080)

Workflow:
  1. Drop images into ./assets/images/
  2. Drop a music file at ./assets/music/background.mp3
  3. Edit ./content.csv with rows: image,text
  4. Run: python make_video.py
  5. Output: ./output/astronomy_video.mp4

Architecture:
  Frames are composed in-process with cv2/numpy/Pillow and streamed via stdin
  into a single ffmpeg encode. There is no intermediate file I/O for frames
  and no clip-graph compositing overhead, so a 30-second 1080p video typically
  renders in well under a minute.
"""

import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
VIDEO_W, VIDEO_H = 1920, 1080
FPS = 30

READING_SPEED = 13           # chars per second; lower = slower, more readable
MIN_DURATION = 5.0
MAX_DURATION = 12.0
TRANSITION_TIME = 1.0        # crossfade duration between slides
TEXT_FADE_IN = 0.8           # text fade-in duration on each slide

KEN_BURNS_ZOOM = 1.12        # 1.0 disables; 1.12 = gentle 12% zoom

FONT_SIZE = 56
LINE_SPACING = 14
TEXT_COLOR = (255, 255, 255)
TEXT_STROKE_COLOR = (0, 0, 0)
TEXT_STROKE_WIDTH = 3
TEXT_MAX_WIDTH_RATIO = 0.85
TEXT_BOTTOM_MARGIN = 90
BACKDROP_PADDING = 30
BACKDROP_OPACITY = 140       # 0-255

MUSIC_FADE_IN = 1.5
MUSIC_FADE_OUT = 2.5
MUSIC_VOLUME = 0.35

# Encode settings — tweak preset/crf for speed-vs-quality
H264_PRESET = "medium"       # "ultrafast" for fastest, "slow" for best quality
H264_CRF = 20                # lower = better quality. 18-23 is the typical range.

BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "assets" / "images"
MUSIC_PATH = BASE_DIR / "assets" / "music" / "background.mp3"
CSV_PATH = BASE_DIR / "content.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "astronomy_video.mp4"

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def find_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    print("[warn] No TrueType font found — falling back to default (lower quality)")
    return ImageFont.load_default()


def find_ffmpeg() -> str:
    """Use the ffmpeg bundled with imageio-ffmpeg if a system one isn't on PATH."""
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("ffmpeg not found. Install with `pip install imageio-ffmpeg` or system ffmpeg.")


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines, current = [], []
    for word in words:
        trial = " ".join(current + [word])
        bbox = font.getbbox(trial)
        if bbox[2] - bbox[0] <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def render_text_overlay(text: str) -> np.ndarray:
    """Renders text + soft backdrop as a full-frame RGBA numpy array (H, W, 4)."""
    font = find_font(FONT_SIZE)
    max_text_width = int(VIDEO_W * TEXT_MAX_WIDTH_RATIO)
    lines = wrap_text(text, font, max_text_width)

    line_widths, line_heights = [], []
    for line in lines:
        bbox = font.getbbox(line)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    text_block_w = max(line_widths) if line_widths else 0
    text_block_h = sum(line_heights) + LINE_SPACING * (len(lines) - 1)

    backdrop_w = text_block_w + BACKDROP_PADDING * 2
    backdrop_h = text_block_h + BACKDROP_PADDING * 2
    backdrop_x = (VIDEO_W - backdrop_w) // 2
    backdrop_y = VIDEO_H - TEXT_BOTTOM_MARGIN - backdrop_h

    canvas = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0, 0, 0, 0))

    backdrop_layer = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0, 0, 0, 0))
    bd_draw = ImageDraw.Draw(backdrop_layer)
    bd_draw.rounded_rectangle(
        [backdrop_x, backdrop_y, backdrop_x + backdrop_w, backdrop_y + backdrop_h],
        radius=20,
        fill=(0, 0, 0, BACKDROP_OPACITY),
    )
    backdrop_layer = backdrop_layer.filter(ImageFilter.GaussianBlur(radius=2))
    canvas = Image.alpha_composite(canvas, backdrop_layer)

    draw = ImageDraw.Draw(canvas)
    cursor_y = backdrop_y + BACKDROP_PADDING
    for line, lw, lh in zip(lines, line_widths, line_heights):
        cursor_x = (VIDEO_W - lw) // 2
        draw.text(
            (cursor_x, cursor_y),
            line,
            font=font,
            fill=TEXT_COLOR,
            stroke_width=TEXT_STROKE_WIDTH,
            stroke_fill=TEXT_STROKE_COLOR,
        )
        cursor_y += lh + LINE_SPACING

    return np.array(canvas)


def prepare_image(path: Path) -> np.ndarray:
    """
    Cover-fits an image to (VIDEO_W * KEN_BURNS_ZOOM) x (VIDEO_H * KEN_BURNS_ZOOM)
    so the most-zoomed-in Ken Burns crop still has native pixel density.
    Returns RGB uint8 (H, W, 3).
    """
    target_w = int(VIDEO_W * KEN_BURNS_ZOOM)
    target_h = int(VIDEO_H * KEN_BURNS_ZOOM)
    target_ratio = target_w / target_h

    img = Image.open(path).convert("RGB")
    src_w, src_h = img.size
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        offset = (src_w - new_w) // 2
        img = img.crop((offset, 0, offset + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        offset = (src_h - new_h) // 2
        img = img.crop((0, offset, src_w, offset + new_h))

    img = img.resize((target_w, target_h), Image.LANCZOS)
    return np.array(img)


# ---------------------------------------------------------------------------
# SLIDE
# ---------------------------------------------------------------------------
class Slide:
    """
    A slide owns:
      - The Ken Burns source image (upscaled, RGB).
      - Two precomputed text composites (BGR, full alpha):
          * `_text_premul_bgr`:    overlay RGB premultiplied by alpha, BGR order
          * `_text_inv_alpha_bgr`: (255 - alpha) broadcast to 3 channels, BGR order
        Each clipped to the text bounding box for a ~10x speedup over full-frame.
      - The text bounding box `(y0, y1, x0, x1)`.

    The hot per-frame path:
      1. Ken Burns crop+resize -> BGR background (cv2, fast).
      2. Inside the bbox: bg[box] = cv2.add(cv2.multiply(bg[box], inv_alpha), premul).
      3. For frames inside the fade-in window we do the slower per-frame blend.
    """

    def __init__(self, image_path: Path, text: str):
        self.image = prepare_image(image_path)
        text_length = len(text)
        self.duration = max(MIN_DURATION, min(MAX_DURATION, text_length / READING_SPEED))
        self.n_frames = int(round(self.duration * FPS))
        self.src_h, self.src_w = self.image.shape[:2]

        # Render the text overlay once (RGBA, full frame size)
        rgba = render_text_overlay(text)              # (H, W, 4) uint8 RGB+A

        # Find the text bounding box so we only blend a small region per frame
        ys, xs = np.where(rgba[..., 3] > 0)
        if len(ys) == 0:
            # No visible text — degenerate; render bg only.
            self._has_text = False
            return
        self._has_text = True
        self._tbox = (int(ys.min()), int(ys.max()) + 1,
                      int(xs.min()), int(xs.max()) + 1)
        y0, y1, x0, x1 = self._tbox

        crop_rgba = rgba[y0:y1, x0:x1]                 # (h, w, 4)
        crop_rgb_bgr = cv2.cvtColor(crop_rgba[..., :3], cv2.COLOR_RGB2BGR)
        alpha = crop_rgba[..., 3:4]                    # (h, w, 1)
        alpha_3 = cv2.cvtColor(alpha[..., 0], cv2.COLOR_GRAY2BGR)   # (h, w, 3)

        # Premultiplied overlay (full-opacity baseline, in BGR)
        self._text_premul_bgr = cv2.multiply(crop_rgb_bgr, alpha_3, scale=1/255)
        # Inverse alpha as 3-channel uint8 (ready to multiply with the BG patch)
        self._text_inv_alpha_bgr = 255 - alpha_3
        # Keep the raw alpha (1-channel) for the fade-in path
        self._text_alpha_1 = alpha                     # (h, w, 1) uint8
        self._text_rgb_bgr = crop_rgb_bgr

    def _composite_text_full_opacity(self, bg_bgr: np.ndarray) -> None:
        """In-place blend of the precomputed full-opacity text into bg_bgr."""
        y0, y1, x0, x1 = self._tbox
        patch = bg_bgr[y0:y1, x0:x1]
        attenuated = cv2.multiply(patch, self._text_inv_alpha_bgr, scale=1/255)
        bg_bgr[y0:y1, x0:x1] = cv2.add(attenuated, self._text_premul_bgr)

    def _composite_text_with_opacity(self, bg_bgr: np.ndarray, opacity: float) -> None:
        """Per-frame blend used during the short fade-in window."""
        if opacity <= 0:
            return
        y0, y1, x0, x1 = self._tbox
        scaled_alpha_1 = (self._text_alpha_1.astype(np.float32) * opacity).astype(np.uint8)
        scaled_alpha_3 = cv2.cvtColor(scaled_alpha_1[..., 0], cv2.COLOR_GRAY2BGR)
        inv = 255 - scaled_alpha_3
        patch = bg_bgr[y0:y1, x0:x1]
        a = cv2.multiply(patch, inv, scale=1/255)
        b = cv2.multiply(self._text_rgb_bgr, scaled_alpha_3, scale=1/255)
        bg_bgr[y0:y1, x0:x1] = cv2.add(a, b)

    def render_frame(self, frame_idx: int) -> np.ndarray:
        """Returns BGR uint8 frame ready for ffmpeg's -pix_fmt bgr24 input."""
        t = frame_idx / FPS
        progress = min(1.0, t / self.duration) if self.duration > 0 else 1.0

        # Ken Burns: crop a shrinking centered window, then resize to output
        zoom = 1.0 + (KEN_BURNS_ZOOM - 1.0) * progress
        crop_w = int(self.src_w / zoom)
        crop_h = int(self.src_h / zoom)
        x0 = (self.src_w - crop_w) // 2
        y0 = (self.src_h - crop_h) // 2
        window = self.image[y0:y0 + crop_h, x0:x0 + crop_w]
        bg_rgb = cv2.resize(window, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_LINEAR)
        bg_bgr = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)

        if self._has_text:
            if TEXT_FADE_IN > 0 and t < TEXT_FADE_IN:
                self._composite_text_with_opacity(bg_bgr, t / TEXT_FADE_IN)
            else:
                self._composite_text_full_opacity(bg_bgr)

        return bg_bgr


# ---------------------------------------------------------------------------
# RENDER PIPELINE
# ---------------------------------------------------------------------------
def stream_frames_to_ffmpeg(slides: list[Slide], silent_video_path: Path):
    """
    Pipe composed BGR frames into ffmpeg over stdin. Slide-to-slide transitions
    are linear crossfades produced by blending the tail of slide N with the
    head of slide N+1.
    """
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{VIDEO_W}x{VIDEO_H}",
        "-pix_fmt", "bgr24",
        "-r", str(FPS),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", H264_PRESET,
        "-crf", str(H264_CRF),
        "-movflags", "+faststart",
        str(silent_video_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    transition_frames = int(round(TRANSITION_TIME * FPS))
    total_written = 0
    t_start = time.time()

    try:
        for i, slide in enumerate(slides):
            is_first = (i == 0)
            is_last = (i == len(slides) - 1)
            next_slide = slides[i + 1] if not is_last else None

            # Head: from `start` (skipping the incoming-fade region already
            # written as part of the previous slide's outgoing crossfade) up to
            # but not including the outgoing-fade region.
            start = 0 if is_first else transition_frames
            end = slide.n_frames - (transition_frames if not is_last else 0)
            for f in range(start, end):
                proc.stdin.write(slide.render_frame(f).tobytes())
                total_written += 1

            # Outgoing crossfade with the next slide's head
            if not is_last:
                for k in range(transition_frames):
                    a = slide.render_frame(end + k)
                    b = next_slide.render_frame(k)
                    alpha = (k + 1) / transition_frames
                    blend = cv2.addWeighted(a, 1.0 - alpha, b, alpha, 0)
                    proc.stdin.write(blend.tobytes())
                    total_written += 1

            elapsed = time.time() - t_start
            fps = total_written / elapsed if elapsed > 0 else 0
            print(f"  slide {i+1}/{len(slides)} done ({total_written} frames, {fps:.1f} fps)")

        proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            err = proc.stderr.read().decode(errors="replace")
            raise SystemExit(f"ffmpeg failed (exit {ret}):\n{err}")
        print(f"[info] Wrote {total_written} frames "
              f"({total_written / FPS:.1f}s) in {time.time() - t_start:.1f}s")
    except BrokenPipeError:
        err = proc.stderr.read().decode(errors="replace")
        raise SystemExit(f"ffmpeg pipe broke:\n{err}")


def mux_audio(silent_video: Path, music: Path, video_duration: float, output: Path):
    """Combine video with looped + faded background music using ffmpeg."""
    ffmpeg = find_ffmpeg()
    if not music.exists():
        print(f"[warn] No music at {music} — copying silent video as final output")
        shutil.copy(silent_video, output)
        return

    fade_out_start = max(0.0, video_duration - MUSIC_FADE_OUT)
    audio_filter = (
        f"aloop=loop=-1:size=2e9,"
        f"atrim=duration={video_duration:.3f},"
        f"volume={MUSIC_VOLUME},"
        f"afade=t=in:st=0:d={MUSIC_FADE_IN},"
        f"afade=t=out:st={fade_out_start:.3f}:d={MUSIC_FADE_OUT}"
    )
    cmd = [
        ffmpeg, "-y",
        "-loglevel", "error",
        "-i", str(silent_video),
        "-i", str(music),
        "-filter_complex", f"[1:a]{audio_filter}[a]",
        "-map", "0:v",
        "-map", "[a]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg mux failed:\n{result.stderr.decode(errors='replace')}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing content.csv at {CSV_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_name = row["image"].strip()
            text = row["text"].strip()
            if not image_name or not text:
                continue
            image_path = IMAGES_DIR / image_name
            if not image_path.exists():
                print(f"[warn] skipping missing image: {image_path}")
                continue
            rows.append((image_path, text))

    if not rows:
        raise SystemExit("No valid rows in content.csv")

    print(f"[info] Preparing {len(rows)} slides...")
    slides = []
    for i, (image_path, text) in enumerate(rows, 1):
        print(f"  [{i}/{len(rows)}] {image_path.name} ({len(text)} chars)")
        slides.append(Slide(image_path, text))

    # Total duration accounts for crossfade overlap (each transition saves TRANSITION_TIME)
    total_duration = sum(s.duration for s in slides) - TRANSITION_TIME * (len(slides) - 1)
    print(f"[info] Total duration: {total_duration:.1f}s")

    silent_path = OUTPUT_DIR / "_silent.mp4"
    print(f"[info] Encoding video...")
    stream_frames_to_ffmpeg(slides, silent_path)

    print(f"[info] Muxing audio...")
    mux_audio(silent_path, MUSIC_PATH, total_duration, OUTPUT_PATH)
    silent_path.unlink(missing_ok=True)

    print(f"[done] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
