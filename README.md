# Astronomy Video Generator

YouTube-ready 1920x1080 astronomy videos from images + text. No ImageMagick required.

## Setup (one-time)

```bash
pip install opencv-python pillow numpy imageio-ffmpeg
```

That's it. `imageio-ffmpeg` ships its own ffmpeg binary, so you don't need to install ffmpeg system-wide. (If you already have system ffmpeg on your PATH, the script will use that instead — slightly faster on Apple Silicon.)

## Folder layout

```
astronomy_video/
├── make_video.py
├── content.csv          ← edit this
├── assets/
│   ├── images/          ← drop your images here
│   └── music/
│       └── background.mp3
└── output/
    └── astronomy_video.mp4   ← generated
```

Images can be any size or aspect ratio — they'll be center-cropped to 16:9. JPEG, PNG, and WebP all work.

## The 2-minute workflow

1. Drop new images into `assets/images/`.
2. Replace `assets/music/background.mp3` if you want a different track.
3. Open `content.csv` and write one row per slide:
   ```csv
   image,text
   nebula.jpg,The Orion Nebula is a stellar nursery 1344 light-years away.
   eclipse.jpg,During a total solar eclipse the Moon perfectly covers the Sun.
   ```
4. Run:
   ```bash
   python make_video.py
   ```
5. Upload `output/astronomy_video.mp4` to YouTube.

## How it works

The pipeline avoids MoviePy's clip-graph compositing (which was the source of your performance and ImageMagick problems) and instead:

1. **Text rendering** is done once per slide with Pillow — no ImageMagick needed.
2. **Ken Burns** zooms each image with `cv2.resize` (200+ fps even on a laptop CPU).
3. **Text composition** uses bbox-clipped premultiplied alpha (~2000 fps) — only the small text region gets blended each frame, not the whole 1080p canvas.
4. **Crossfades** between slides are computed in-process with `cv2.addWeighted`.
5. Composed BGR frames are streamed straight into a single ffmpeg `libx264` encode over stdin — no temporary frame files on disk.
6. **Audio** is muxed in a second pass: ffmpeg loops, trims, and fades the music to fit the video.

### Real-world render speed

On a recent MacBook (M1+) you can expect roughly:

| Video length | Render time (preset=medium) | Render time (preset=ultrafast) |
| ---:         | ---:                        | ---:                           |
| 30 seconds   | ~15-20 s                    | ~6-8 s                         |
| 2 minutes    | ~50-70 s                    | ~20-30 s                       |
| 10 minutes   | ~5-7 min                    | ~2-3 min                       |

Older / single-core machines will be ~3-4x slower.

## Tweaking the look

All knobs are at the top of `make_video.py`:

| Variable             | What it does                                         | Default |
| ---                  | ---                                                  | ---     |
| `READING_SPEED`      | Chars / second. Lower = slower, more readable.       | `13`    |
| `MIN_DURATION`       | Floor on per-slide length (seconds).                 | `5.0`   |
| `MAX_DURATION`       | Ceiling on per-slide length (seconds).               | `12.0`  |
| `KEN_BURNS_ZOOM`     | Zoom factor over a slide. `1.0` = static.            | `1.12`  |
| `TRANSITION_TIME`    | Crossfade between slides (seconds).                  | `1.0`   |
| `TEXT_FADE_IN`       | Text fade-in on each slide (seconds).                | `0.8`   |
| `FONT_SIZE`          | Text point size at 1920x1080.                        | `56`    |
| `BACKDROP_OPACITY`   | 0-255. Higher = darker text bar.                     | `140`   |
| `MUSIC_VOLUME`       | 0.0-1.0. `0.35` is comfortable background level.     | `0.35`  |
| `H264_PRESET`        | `ultrafast`/`fast`/`medium`/`slow`. Speed vs quality.| `medium`|
| `H264_CRF`           | Quality. Lower = better. 18-23 is the typical range. | `20`    |

For fast iteration, set `H264_PRESET = "ultrafast"` while you're testing content; switch back to `"medium"` (or `"slow"` for archival) for the final upload.

## Output spec

- 1920x1080 @ 30 fps
- H.264, yuv420p, CRF 20 (~5-8 Mbps on real footage)
- AAC stereo @ 192 kbps
- `+faststart` for instant playback on YouTube

This matches YouTube's recommended 1080p upload spec, so YouTube will re-encode without quality loss artifacts.

## Common tweaks

**Add a title/outro slide:** add a row to `content.csv` with a black image and a short title like "ASTRONOMY • EPISODE 12". The same slide template handles it.

**Vertical Shorts (1080×1920):** change `VIDEO_W, VIDEO_H = 1080, 1920` at the top. The cover-fit code handles portrait images correctly.

**Different fonts:** edit `FONT_CANDIDATES` at the top to point at your `.ttf` of choice. Bold fonts read best with a video background.

**Different aspect images per slide:** the cover-crop is centered. If you have a portrait photo where the subject is at the top, pre-crop the image yourself before dropping it in.
