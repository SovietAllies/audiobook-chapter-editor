#!/usr/bin/env python3
"""
Audiobook Editor (GTK3) — Chapters + Cover Art + Metadata editor for M4B (and MP3→M4B) using ffmpeg.

Copyright (C) 2026 SovietAllies

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Credits:
- Developed with the help of ChatGPT (GPT-5.2).

Overview:
- Chapters tab: editable chapter list (HH:MM:SS[.mmm]), ffplay preview, silence-based chapter detection.
- Cover art: auto-detect embedded art (attached_pic), preview it, optionally embed new jpg/png art.
- Metadata tab: generic editable table that loads *all* ffprobe-visible metadata (global + stream-scoped).
- MP3 → M4B conversion: optional conversion with AAC VBR/CBR.

Requirements:
- ffmpeg, ffprobe, ffplay available in PATH
- GTK3 Python bindings (PyGObject)

Ubuntu/Debian:
  sudo apt install ffmpeg python3-gi gir1.2-gtk-3.0
"""

# -----------------------------
# GTK (PyGObject) imports
# -----------------------------
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib, GdkPixbuf

# -----------------------------
# Standard library imports
# -----------------------------
import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from typing import Optional, List, Tuple, Dict


# -----------------------------
# Regex helpers
# -----------------------------

# Accept HH:MM:SS or HH:MM:SS.mmm (mmm optional)
TS_RE = re.compile(r"^\s*(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?\s*$")

# ffmpeg silencedetect often outputs:
#   silence_end: 123.456 | silence_duration: 3.000
SILENCE_END_RE = re.compile(
    r"silence_end:\s*([0-9]*\.?[0-9]+)\s*\|\s*silence_duration:\s*([0-9]*\.?[0-9]+)"
)


# ==========================================================
# Time helpers
# ==========================================================

def parse_ts_to_ms(ts: str) -> int:
    """
    Convert a GUI timestamp string into milliseconds.
    Accepts:
      - HH:MM:SS
      - HH:MM:SS.mmm
    """
    m = TS_RE.match(ts.strip())
    if not m:
        raise ValueError(f"Bad time '{ts}'. Use HH:MM:SS or HH:MM:SS.mmm")

    hh, mm, ss, frac = m.groups()
    hh = int(hh)
    mm = int(mm)
    ss = int(ss)

    # Basic bounds check (helps catch typos like 00:99:00)
    if mm > 59 or ss > 59:
        raise ValueError(f"Bad time '{ts}' (mm/ss must be 00-59)")

    # Fractional part is milliseconds; normalize to 3 digits.
    ms = int((frac or "0").ljust(3, "0"))

    return (((hh * 60 + mm) * 60 + ss) * 1000) + ms


def ms_to_ts(ms: int) -> str:
    """
    Convert milliseconds into HH:MM:SS or HH:MM:SS.mmm (if needed).
    """
    if ms < 0:
        ms = 0

    seconds, milli = divmod(ms, 1000)
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)

    if milli:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}.{milli:03d}"
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def file_ext_lower(path: str) -> str:
    """Return lowercase file extension including '.', e.g. '.m4b'."""
    return os.path.splitext(path)[1].lower().strip()


# ==========================================================
# ffprobe helpers (reading chapters, tags, art)
# ==========================================================

def ffprobe_json(args: List[str]) -> dict:
    """
    Run ffprobe and return parsed JSON.
    We use -v error to minimize noise.
    """
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-print_format", "json"] + args,
        text=True
    )
    return json.loads(out)


def get_duration_ms(path: str) -> int:
    """Read duration from ffprobe and convert to ms."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        text=True,
    ).strip()
    return int(round(float(out) * 1000))


def probe_chapters(path: str) -> List[Tuple[int, str]]:
    """
    Read chapters from container using ffprobe -show_chapters.
    Return list of (start_ms, title) sorted by time.
    """
    try:
        data = ffprobe_json(["-show_chapters", path])
        chapters = data.get("chapters", []) or []
        res: List[Tuple[int, str]] = []

        for idx, ch in enumerate(chapters, start=1):
            start_s = float(ch.get("start_time", "0"))
            tags = ch.get("tags", {}) or {}
            title = tags.get("title") or tags.get("TITLE") or f"Chapter {idx}"
            res.append((int(round(start_s * 1000)), str(title)))

        res.sort(key=lambda x: x[0])
        return res
    except Exception:
        return []


def probe_format_tags(path: str) -> Dict[str, str]:
    """
    Read global/container tags from the format section.
    Returns {key: value} exactly as ffprobe provides (case preserved).
    """
    try:
        data = ffprobe_json(["-show_format", path])
        tags = ((data.get("format") or {}).get("tags") or {})
        out: Dict[str, str] = {}
        for k, v in tags.items():
            if v is None:
                continue
            out[str(k)] = str(v)
        return out
    except Exception:
        return {}


def probe_stream_tags_with_scopes(path: str) -> List[Tuple[str, Dict[str, str]]]:
    """
    Return per-stream tags with an ffmpeg scope label.

    ffmpeg stream metadata scoping looks like:
      -metadata:s:a:0 key=value   (audio stream #0)
      -metadata:s:v:0 key=value   (video stream #0)

    ffprobe stream objects have codec_type, but not always a stable "a:0" index.
    So we compute per-type indexes by counting streams in input order.
    """
    try:
        data = ffprobe_json(["-show_streams", path])
        streams = data.get("streams", []) or []

        counters = {"audio": 0, "video": 0, "subtitle": 0, "data": 0}
        type_to_letter = {"audio": "a", "video": "v", "subtitle": "s", "data": "d"}

        out: List[Tuple[str, Dict[str, str]]] = []

        for st in streams:
            ctype = st.get("codec_type")
            if ctype not in counters:
                continue

            letter = type_to_letter[ctype]
            idx = counters[ctype]
            counters[ctype] += 1

            tags = (st.get("tags") or {})
            if not tags:
                continue

            norm: Dict[str, str] = {}
            for k, v in tags.items():
                if v is None:
                    continue
                norm[str(k)] = str(v)

            scope = f"s:{letter}:{idx}"
            out.append((scope, norm))

        return out
    except Exception:
        return []


def has_video_stream(path: str) -> bool:
    """True if there is any video stream (often used to hold attached_pic artwork)."""
    try:
        data = ffprobe_json(["-show_streams", "-select_streams", "v", path])
        return bool((data.get("streams", []) or []))
    except Exception:
        return False


def find_attached_pic_stream_index(path: str) -> Optional[int]:
    """
    Find stream index (absolute stream index) for embedded cover art.
    Cover art in MP4/M4B often appears as a video stream with disposition attached_pic=1.
    """
    try:
        data = ffprobe_json(["-show_streams", path])
        for st in data.get("streams", []) or []:
            if st.get("codec_type") == "video":
                disp = st.get("disposition") or {}
                if int(disp.get("attached_pic", 0)) == 1:
                    return int(st.get("index"))
    except Exception:
        pass
    return None


def extract_embedded_art_to_png(path: str, out_png_path: str) -> bool:
    """
    Extract embedded cover art to a PNG file for reliable GTK preview.

    Why re-encode as PNG?
    - Embedded art codecs vary, and GTK preview is simplest if we load a standard image.
    """
    idx = find_attached_pic_stream_index(path)
    if idx is None:
        return False

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-map", f"0:{idx}",
        "-frames:v", "1",
        "-c:v", "png",
        out_png_path
    ]
    try:
        subprocess.check_call(cmd)
        return os.path.exists(out_png_path) and os.path.getsize(out_png_path) > 0
    except Exception:
        return False


# ==========================================================
# ffmpeg helpers (write, scan, conversion)
# ==========================================================

def run_and_stream_output(cmd: List[str], on_line) -> int:
    """
    Run command and stream combined stdout/stderr lines to on_line callback.
    Used to keep UI log updated with ffmpeg progress/errors.
    """
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert p.stdout is not None
    for line in p.stdout:
        on_line(line)
    return p.wait()


def build_ffmeta(starts_titles: List[Tuple[int, str]], duration_ms: int) -> str:
    """
    Build an FFMETADATA1 file content for chapters.

    TIMEBASE=1/1000 means START/END are ms.
    END of a chapter is the START of the next chapter, last chapter ends at duration.
    """
    lines = [";FFMETADATA1"]
    for i, (start, title) in enumerate(starts_titles):
        end = starts_titles[i + 1][0] if i + 1 < len(starts_titles) else duration_ms
        if end <= start:
            raise ValueError(f"Chapter '{title}' has end <= start ({ms_to_ts(end)} <= {ms_to_ts(start)})")
        title = title.replace("\n", " ").strip()
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start}",
            f"END={end}",
            f"title={title}",
            "",
        ]
    return "\n".join(lines).strip() + "\n"


def detect_chapters_from_silence(path: str, noise_db: float, min_dur: float, mark_offset: float = 1.0) -> List[int]:
    """
    Detect silence regions and create candidate chapter markers.

    Rule:
      If silence lasts >= min_dur seconds, mark a chapter at (silence_end - mark_offset).

    Default mark_offset=1.0 implements:
  “if the silence is 2.5s+ then the last second should be marked as chapter”.
    """
    cmd = [
        "ffmpeg", "-hide_banner",
        "-i", path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
        "-f", "null", "-"
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    starts_ms: List[int] = []

    assert p.stdout is not None
    for line in p.stdout:
        m_end = SILENCE_END_RE.search(line)
        if m_end:
            silence_end = float(m_end.group(1))
            silence_dur = float(m_end.group(2))
            if silence_dur >= min_dur:
                t = max(0.0, silence_end - mark_offset)
                starts_ms.append(int(round(t * 1000)))

    p.wait()
    return sorted(set(starts_ms))


def parse_bitrate_field_to_bps(s: str) -> Optional[int]:
    """
    Parse bitrate strings to bits-per-second.
    Accepts:
      "128000", "128k", "256kbps", "n/a", etc.
    """
    if not s:
        return None
    s = s.strip().lower()
    if s == "n/a":
        return None
    if s.isdigit():
        return int(s)

    m = re.match(r"^(\d+(?:\.\d+)?)\s*([kmg])?b?ps?$", s)
    if not m:
        return None

    val = float(m.group(1))
    unit = m.group(2)
    if unit == "k":
        return int(val * 1000)
    if unit == "m":
        return int(val * 1_000_000)
    if unit == "g":
        return int(val * 1_000_000_000)
    return int(val)


def detect_mp3_bitrate_bps(path: str) -> Optional[int]:
    """
    Best-effort MP3 bitrate detection so AAC CBR can match source roughly.
    """
    try:
        data = ffprobe_json(["-show_streams", "-show_format", path])

        # Prefer audio stream bitrate
        for st in data.get("streams", []) or []:
            if st.get("codec_type") == "audio":
                br = parse_bitrate_field_to_bps(str(st.get("bit_rate") or ""))
                if br and br > 0:
                    return br

        # Fallback to container bitrate
        brf = parse_bitrate_field_to_bps(str((data.get("format") or {}).get("bit_rate") or ""))
        if brf and brf > 0:
            return brf
    except Exception:
        pass

    return None


def bps_to_aac_bitrate_arg(bps: int) -> str:
    """
    Convert bps into an ffmpeg-friendly bitrate string, with mild clamping.
    """
    kbps = int(round(bps / 1000))
    kbps = max(48, min(320, kbps))
    return f"{kbps}k"


def image_codec_for_file(path: str) -> str:
    """
    Choose a safe codec for embedding cover art into MP4/M4B:
      jpg/jpeg -> mjpeg
      png      -> png
    """
    ext = file_ext_lower(path)
    if ext in (".jpg", ".jpeg"):
        return "mjpeg"
    if ext == ".png":
        return "png"
    return "mjpeg"


# ==========================================================
# GTK GUI class
# ==========================================================

class ChapterGUI(Gtk.Window):
    """
    Main application window.

    A few important state variables:
      self.input_path          currently loaded file
      self.player_proc         ffplay subprocess (or None)
      self.detected_silence_ms cached silence candidates
      self.job_running         True while background ffmpeg job runs
      self.user_cover_path     user-selected cover art to embed (optional)
      self.embedded_art_temp   temp extracted art preview path (optional)
    """

    def __init__(self):
        super().__init__(title="Audiobook Editor — Chapters / Cover Art / Metadata")
        self.set_border_width(10)
        self.set_default_size(1260, 820)

        # --- runtime state ---
        self.input_path: Optional[str] = None
        self.player_proc: Optional[subprocess.Popen] = None
        self.detected_silence_ms: List[int] = []
        self.job_running = False

        self.user_cover_path: Optional[str] = None
        self.embedded_art_temp: Optional[str] = None

        # Log pane visibility state
        self.log_visible = True
        self._log_prev_pos: Optional[int] = None

        # Main UI layout: vertical paned (top content + bottom log)
        self.vpaned = Gtk.Paned.new(Gtk.Orientation.VERTICAL)
        self.add(self.vpaned)

        # Top content area (file controls + notebook)
        top_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.vpaned.pack1(top_box, resize=True, shrink=False)

        # File chooser + status + write
        row = Gtk.Box(spacing=8)
        top_box.pack_start(row, False, False, 0)

        self.btn_open = Gtk.Button(label="Choose .m4b or .mp3…")
        self.btn_open.connect("clicked", self.on_choose_file)
        row.pack_start(self.btn_open, False, False, 0)

        self.lbl_file = Gtk.Label(label="No file selected")
        self.lbl_file.set_xalign(0)
        row.pack_start(self.lbl_file, True, True, 0)

        self.btn_write = Gtk.Button(label="Write → new .m4b…")
        self.btn_write.connect("clicked", self.on_write)
        row.pack_start(self.btn_write, False, False, 0)

        # Tabs
        self.notebook = Gtk.Notebook()
        top_box.pack_start(self.notebook, True, True, 0)

        self.tab_chapters = self._build_tab_chapters()
        self.tab_metadata = self._build_tab_metadata()

        self.notebook.append_page(self.tab_chapters, Gtk.Label(label="Chapters"))
        self.notebook.append_page(self.tab_metadata, Gtk.Label(label="Metadata"))

        # Bottom log panel (footer always visible + hide/show log content)
        #
        # IMPORTANT: The toggle button lives in the footer, which stays visible even when the
        # log content is hidden. This prevents the “can't bring log back” problem.
        bottom_outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.vpaned.pack2(bottom_outer, resize=False, shrink=False)

        # Persistent footer bar (always visible)
        footer = Gtk.Box(spacing=8)
        bottom_outer.pack_start(footer, False, False, 0)

        self.btn_toggle_log = Gtk.Button(label="Hide Log")
        self.btn_toggle_log.connect("clicked", self.on_toggle_log)
        footer.pack_start(self.btn_toggle_log, False, False, 0)

        # Spacer/filler area (reserved for future buttons)
        footer.pack_start(Gtk.Label(label=""), True, True, 0)

        # Log content area (this is what gets hidden/shown)
        self.log_content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        bottom_outer.pack_start(self.log_content_box, True, True, 0)

        log_header = Gtk.Box(spacing=8)
        self.log_content_box.pack_start(log_header, False, False, 0)
        log_header.pack_start(Gtk.Label(label="Log"), False, False, 0)

        self.log = Gtk.TextView(editable=False)
        self.log.set_monospace(True)
        sc = Gtk.ScrolledWindow()
        sc.set_vexpand(True)
        sc.add(self.log)
        self.log_content_box.pack_start(sc, True, True, 0)

        # Set initial divider position after GTK lays out widgets
        GLib.idle_add(self.vpaned.set_position, 580)

        # Stop ffplay + cleanup temp files when closing
        self.connect("delete-event", self.on_window_delete)

        # Seed initial tables
        self.reset_chapters_table()
        self.reset_metadata_table()

        # Initial state
        self.update_ui_state()

    # ==========================================================
    # Logging helpers
    # ==========================================================

    def append_log(self, msg: str):
        """
        Append log text to the TextView and scroll to bottom.
        Must be called on GTK main thread.
        """
        buf = self.log.get_buffer()
        buf.insert(buf.get_end_iter(), msg)
        mark = buf.create_mark(None, buf.get_end_iter(), False)
        self.log.scroll_to_mark(mark, 0.0, True, 0.0, 1.0)

    def ui_log(self, msg: str):
        """
        Thread-safe logger: schedule append_log on GTK main loop.
        Use this from worker threads.
        """
        GLib.idle_add(self.append_log, msg)

    def on_toggle_log(self, _):
        """
        Toggle visibility of the *log content* while keeping the footer bar visible.
        The same button switches between Hide Log / Show Log.
        """
        # If widgets haven't been created yet, bail out safely
        if not hasattr(self, "log_content_box"):
            return

        if self.log_visible:
            # Remember current split so we can restore it when showing again
            self._log_prev_pos = self.vpaned.get_position()

            # Hide log content (footer stays)
            self.log_content_box.set_visible(False)

            # Nudge divider down so the bottom area becomes small (just the footer)
            # Using allocated height keeps this proportional across window sizes.
            try:
                GLib.idle_add(self.vpaned.set_position, max(0, self.get_allocated_height() - 44))
            except Exception:
                pass

            self.btn_toggle_log.set_label("Show Log")
            self.log_visible = False
        else:
            # Show log content again
            self.log_content_box.set_visible(True)

            # Restore previous split position if we saved one
            if self._log_prev_pos is not None:
                self.vpaned.set_position(self._log_prev_pos)

            self.btn_toggle_log.set_label("Hide Log")
            self.log_visible = True

    def player_running(self) -> bool:
        """Return True if ffplay exists and is still alive."""
        return self.player_proc is not None and self.player_proc.poll() is None

    def update_ui_state(self):
        """
        Central place to enable/disable widgets based on:
        - whether a file is loaded
        - whether a background job is running
        - whether silence candidates exist
        - whether input is mp3 (conversion controls)
        """
        has_file = bool(self.input_path)
        is_mp3 = has_file and file_ext_lower(self.input_path) == ".mp3"
        detected = len(self.detected_silence_ms) > 0

        if self.job_running:
            # Prevent conflicting actions while ffmpeg is busy
            self.btn_open.set_sensitive(False)
            self.btn_write.set_sensitive(False)

            # Chapters tab busy-disables
            self.btn_silence.set_sensitive(False)
            self.btn_append_detected.set_sensitive(False)
            self.btn_replace_detected.set_sensitive(False)
            self.btn_log_all_detected.set_sensitive(False)
            self.btn_choose_art.set_sensitive(False)
            self.btn_clear_art.set_sensitive(False)
            self.chk_embed_art.set_sensitive(False)

            # Metadata tab busy-disables
            self.btn_meta_add.set_sensitive(False)
            self.btn_meta_remove.set_sensitive(False)
            self.btn_meta_sort.set_sensitive(False)
            self.btn_meta_clear.set_sensitive(False)
            self.btn_apply_meta.set_sensitive(False)
            self.btn_reload_meta.set_sensitive(False)
            self.chk_overwrite_meta.set_sensitive(False)
        else:
            # File selection should be available once jobs are done
            self.btn_open.set_sensitive(True)
            self.btn_write.set_sensitive(has_file)

            # Silence scan available when file loaded
            self.btn_silence.set_sensitive(has_file)

            # Only enable “use detected” buttons if we have candidates
            self.btn_append_detected.set_sensitive(detected)
            self.btn_replace_detected.set_sensitive(detected)
            self.btn_log_all_detected.set_sensitive(detected)

            # Art controls
            self.btn_choose_art.set_sensitive(has_file)
            self.btn_clear_art.set_sensitive(bool(self.user_cover_path))
            self.chk_embed_art.set_sensitive(has_file)

            # Metadata controls
            self.btn_meta_add.set_sensitive(has_file)
            self.btn_meta_remove.set_sensitive(has_file)
            self.btn_meta_sort.set_sensitive(has_file)
            self.btn_meta_clear.set_sensitive(has_file)
            self.btn_apply_meta.set_sensitive(has_file)
            self.btn_reload_meta.set_sensitive(has_file)
            self.chk_overwrite_meta.set_sensitive(has_file)

        # Playback controls are usable whenever file exists
        for w in (self.btn_playstop, self.btn_m10, self.btn_m5, self.btn_p5, self.btn_p10):
            w.set_sensitive(has_file)

        # Keep button label consistent with process state
        self.btn_playstop.set_label("Stop" if self.player_running() else "Play")

        # MP3 conversion controls only apply to mp3 inputs
        for w in (self.chk_convert_mp3, self.rb_cbr, self.rb_vbr, self.aac_bitrate, self.aac_q):
            w.set_sensitive(is_mp3 and (not self.job_running))

        # Ensure correct enablement between CBR/VBR inputs
        self.on_codec_mode_changed(None)

    # ==========================================================
    # Build tabs
    # ==========================================================

    def _build_tab_chapters(self) -> Gtk.Widget:
        """
        Chapters tab layout:
          Left: cover art preview + choose/clear + embed checkbox
          Right: chapter actions + playback + silence scan + mp3 conversion + chapter table
        """
        outer = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        outer.set_border_width(6)

        # ---- Cover art panel ----
        art_frame = Gtk.Frame(label="Cover Art (embedded / to embed)")
        outer.pack_start(art_frame, False, False, 0)

        art_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        art_box.set_border_width(8)
        art_frame.add(art_box)

        self.art_image = Gtk.Image()
        art_box.pack_start(self.art_image, False, False, 0)

        self.art_label = Gtk.Label(label="No cover art loaded")
        self.art_label.set_xalign(0)
        art_box.pack_start(self.art_label, False, False, 0)

        art_btns = Gtk.Box(spacing=6)
        art_box.pack_start(art_btns, False, False, 0)

        self.btn_choose_art = Gtk.Button(label="Choose Art (jpg/png)…")
        self.btn_choose_art.connect("clicked", self.on_choose_art)
        art_btns.pack_start(self.btn_choose_art, False, False, 0)

        self.btn_clear_art = Gtk.Button(label="Clear Chosen Art")
        self.btn_clear_art.connect("clicked", self.on_clear_chosen_art)
        art_btns.pack_start(self.btn_clear_art, False, False, 0)

        self.chk_embed_art = Gtk.CheckButton(label="Embed chosen art on write")
        self.chk_embed_art.set_active(True)
        art_box.pack_start(self.chk_embed_art, False, False, 0)

        # ---- Right side: chapters + controls ----
        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        outer.pack_start(right, True, True, 0)

        # Chapter list actions
        controls = Gtk.Box(spacing=8)
        right.pack_start(controls, False, False, 0)

        self.btn_add = Gtk.Button(label="Add")
        self.btn_add.connect("clicked", lambda _ : self.on_add())
        controls.pack_start(self.btn_add, False, False, 0)

        self.btn_remove = Gtk.Button(label="Remove Selected")
        self.btn_remove.connect("clicked", lambda _ : self.on_remove())
        controls.pack_start(self.btn_remove, False, False, 0)

        self.btn_sort = Gtk.Button(label="Sort by Time")
        self.btn_sort.connect("clicked", lambda _ : self.on_sort())
        controls.pack_start(self.btn_sort, False, False, 0)

        self.btn_clear = Gtk.Button(label="Clear Chapters")
        self.btn_clear.connect("clicked", lambda _ : self.reset_chapters_table())
        controls.pack_start(self.btn_clear, False, False, 0)

        # Playback controls
        play = Gtk.Box(spacing=8)
        right.pack_start(play, False, False, 0)

        self.btn_playstop = Gtk.Button(label="Play")
        self.btn_playstop.connect("clicked", self.on_playstop_toggle)
        play.pack_start(self.btn_playstop, False, False, 0)

        self.btn_m10 = Gtk.Button(label="Play -10s")
        self.btn_m10.connect("clicked", lambda _ : self.play_from_selected(-10))
        play.pack_start(self.btn_m10, False, False, 0)

        self.btn_m5 = Gtk.Button(label="Play -5s")
        self.btn_m5.connect("clicked", lambda _ : self.play_from_selected(-5))
        play.pack_start(self.btn_m5, False, False, 0)

        self.btn_p5 = Gtk.Button(label="Play +5s")
        self.btn_p5.connect("clicked", lambda _ : self.play_from_selected(+5))
        play.pack_start(self.btn_p5, False, False, 0)

        self.btn_p10 = Gtk.Button(label="Play +10s")
        self.btn_p10.connect("clicked", lambda _ : self.play_from_selected(+10))
        play.pack_start(self.btn_p10, False, False, 0)

        # Silence detection controls
        silence = Gtk.Box(spacing=8)
        right.pack_start(silence, False, False, 0)

        silence.pack_start(Gtk.Label(label="Silence detect:"), False, False, 0)

        self.noise_entry = Gtk.Entry()
        self.noise_entry.set_text("-35")
        self.noise_entry.set_width_chars(6)
        silence.pack_start(Gtk.Label(label="noise dB"), False, False, 0)
        silence.pack_start(self.noise_entry, False, False, 0)

        self.dur_entry = Gtk.Entry()
        self.dur_entry.set_text("2.5")  # default requested
        self.dur_entry.set_width_chars(6)
        silence.pack_start(Gtk.Label(label="min seconds"), False, False, 0)
        silence.pack_start(self.dur_entry, False, False, 0)

        self.btn_silence = Gtk.Button(label="Scan (silence end−1s)")
        self.btn_silence.connect("clicked", self.on_silence_detect)
        silence.pack_start(self.btn_silence, False, False, 0)

        self.btn_append_detected = Gtk.Button(label="Append ALL")
        self.btn_append_detected.connect("clicked", self.on_append_detected)
        silence.pack_start(self.btn_append_detected, False, False, 0)

        self.btn_replace_detected = Gtk.Button(label="Replace with ALL")
        self.btn_replace_detected.connect("clicked", self.on_replace_with_detected)
        silence.pack_start(self.btn_replace_detected, False, False, 0)

        self.btn_log_all_detected = Gtk.Button(label="Log ALL")
        self.btn_log_all_detected.connect("clicked", self.on_log_all_detected)
        silence.pack_start(self.btn_log_all_detected, False, False, 0)

        # MP3 conversion options
        conv = Gtk.Box(spacing=10)
        right.pack_start(conv, False, False, 0)

        self.chk_convert_mp3 = Gtk.CheckButton(label="If input is MP3: convert to M4B then add chapters")
        self.chk_convert_mp3.set_active(True)
        conv.pack_start(self.chk_convert_mp3, False, False, 0)

        # AAC mode selection
        mode_box = Gtk.Box(spacing=8)
        right.pack_start(mode_box, False, False, 0)

        self.rb_cbr = Gtk.RadioButton.new_with_label_from_widget(None, "AAC CBR (bitrate)")
        self.rb_vbr = Gtk.RadioButton.new_with_label_from_widget(self.rb_cbr, "AAC VBR (quality)")
        self.rb_vbr.set_active(True)

        self.rb_cbr.connect("toggled", self.on_codec_mode_changed)
        self.rb_vbr.connect("toggled", self.on_codec_mode_changed)

        mode_box.pack_start(self.rb_cbr, False, False, 0)
        mode_box.pack_start(self.rb_vbr, False, False, 0)

        # CBR bitrate entry
        cbr_row = Gtk.Box(spacing=6)
        right.pack_start(cbr_row, False, False, 0)
        cbr_row.pack_start(Gtk.Label(label="CBR bitrate:"), False, False, 0)

        self.aac_bitrate = Gtk.Entry()
        self.aac_bitrate.set_text("auto")  # auto-match mp3 bitrate if possible
        self.aac_bitrate.set_width_chars(8)
        cbr_row.pack_start(self.aac_bitrate, False, False, 0)
        cbr_row.pack_start(Gtk.Label(label="(e.g. 128k, 256k, or auto)"), False, False, 0)

        # VBR quality control
        vbr_row = Gtk.Box(spacing=6)
        right.pack_start(vbr_row, False, False, 0)
        vbr_row.pack_start(Gtk.Label(label="VBR quality -q:a:"), False, False, 0)

        self.aac_q = Gtk.SpinButton()
        self.aac_q.set_adjustment(Gtk.Adjustment(2, 0, 5, 1, 1, 0))
        self.aac_q.set_value(2)
        vbr_row.pack_start(self.aac_q, False, False, 0)
        vbr_row.pack_start(Gtk.Label(label="(lower is better; try 2 or 3)"), False, False, 0)

        # Chapters table (ListStore + TreeView)
        self.store = Gtk.ListStore(str, str)  # timestamp, title
        self.tree = Gtk.TreeView(model=self.store)
        self.tree.get_selection().set_mode(Gtk.SelectionMode.SINGLE)

        # Timestamp column (editable, validated)
        r_time = Gtk.CellRendererText(editable=True)
        r_time.connect("edited", self.on_cell_edited, 0)
        self.tree.append_column(Gtk.TreeViewColumn("Start (HH:MM:SS)", r_time, text=0))

        # Title column (editable)
        r_title = Gtk.CellRendererText(editable=True)
        r_title.connect("edited", self.on_cell_edited, 1)
        col_title = Gtk.TreeViewColumn("Title", r_title, text=1)
        col_title.set_expand(True)
        self.tree.append_column(col_title)

        sc_table = Gtk.ScrolledWindow()
        sc_table.set_hexpand(True)
        sc_table.set_vexpand(True)
        sc_table.add(self.tree)
        right.pack_start(sc_table, True, True, 0)

        return outer

    def _build_tab_metadata(self) -> Gtk.Widget:
        """
        Metadata tab is a generic key/value editor with scoping.

        Scope examples:
          g      -> global tags
          s:a:0  -> audio stream 0 tags
          s:v:0  -> video stream 0 tags

        This mirrors how the chapters table works: read from file into a table,
        let the user edit it freely, then apply.
        """
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        outer.set_border_width(10)

        info = Gtk.Label(
            label="Metadata loads automatically when a file is selected.\n"
                  "Edit Scope/Key/Value and click Apply Metadata.\n"
                  "Scopes: g (global) or s:a:0 / s:v:0 / s:s:0 etc."
        )
        info.set_xalign(0)
        outer.pack_start(info, False, False, 0)

        # Row of metadata actions
        btns = Gtk.Box(spacing=8)
        outer.pack_start(btns, False, False, 0)

        self.btn_meta_add = Gtk.Button(label="Add")
        self.btn_meta_add.connect("clicked", self.on_meta_add)
        btns.pack_start(self.btn_meta_add, False, False, 0)

        self.btn_meta_remove = Gtk.Button(label="Remove Selected")
        self.btn_meta_remove.connect("clicked", self.on_meta_remove)
        btns.pack_start(self.btn_meta_remove, False, False, 0)

        self.btn_meta_sort = Gtk.Button(label="Sort")
        self.btn_meta_sort.connect("clicked", self.on_meta_sort)
        btns.pack_start(self.btn_meta_sort, False, False, 0)

        self.btn_meta_clear = Gtk.Button(label="Clear")
        self.btn_meta_clear.connect("clicked", self.on_meta_clear)
        btns.pack_start(self.btn_meta_clear, False, False, 0)

        btns.pack_start(Gtk.Label(label="   "), False, False, 0)

        self.chk_overwrite_meta = Gtk.CheckButton(label="Overwrite original file when applying metadata")
        self.chk_overwrite_meta.set_active(False)
        btns.pack_start(self.chk_overwrite_meta, False, False, 0)

        self.btn_apply_meta = Gtk.Button(label="Apply Metadata")
        self.btn_apply_meta.connect("clicked", self.on_apply_metadata)
        btns.pack_start(self.btn_apply_meta, False, False, 0)

        self.btn_reload_meta = Gtk.Button(label="Reload Metadata")
        self.btn_reload_meta.connect("clicked", lambda _ : self.load_metadata_into_table())
        btns.pack_start(self.btn_reload_meta, False, False, 0)

        # Table of metadata rows: scope, key, value
        self.meta_store = Gtk.ListStore(str, str, str)
        self.meta_tree = Gtk.TreeView(model=self.meta_store)
        self.meta_tree.get_selection().set_mode(Gtk.SelectionMode.SINGLE)

        # Scope column
        r_scope = Gtk.CellRendererText(editable=True)
        r_scope.connect("edited", self.on_meta_cell_edited, 0)
        col_scope = Gtk.TreeViewColumn("Scope (g / s:a:0 / s:v:0)", r_scope, text=0)
        col_scope.set_expand(False)
        self.meta_tree.append_column(col_scope)

        # Key column
        r_key = Gtk.CellRendererText(editable=True)
        r_key.connect("edited", self.on_meta_cell_edited, 1)
        col_key = Gtk.TreeViewColumn("Key", r_key, text=1)
        col_key.set_expand(True)
        self.meta_tree.append_column(col_key)

        # Value column
        r_val = Gtk.CellRendererText(editable=True)
        r_val.connect("edited", self.on_meta_cell_edited, 2)
        col_val = Gtk.TreeViewColumn("Value", r_val, text=2)
        col_val.set_expand(True)
        self.meta_tree.append_column(col_val)

        sc = Gtk.ScrolledWindow()
        sc.set_hexpand(True)
        sc.set_vexpand(True)
        sc.add(self.meta_tree)
        outer.pack_start(sc, True, True, 0)

        return outer

    # ==========================================================
    # “New file loaded” reset helpers
    # ==========================================================

    def reset_for_new_file(self):
        """
        When a new file is selected:
        - stop any ongoing playback
        - clear chapters (then auto-load if present)
        - clear silence candidates
        - clear user-chosen art (so old preview doesn't stick)
        - clear metadata table (then auto-load)
        """
        self.stop_player()

        # Clear chapters + silence results
        self.reset_chapters_table()
        self.detected_silence_ms = []

        # Clear chosen cover art; embedded preview will be refreshed immediately after
        self.user_cover_path = None
        self.art_image.clear()
        self.art_label.set_text("Loading embedded art…")

        # Clear metadata (then load)
        self.reset_metadata_table()

    # ==========================================================
    # Chapters table logic
    # ==========================================================

    def reset_chapters_table(self):
        """Clear chapters table and seed a basic first chapter."""
        self.store.clear()
        self.store.append(["00:00:00", "Start"])

    def on_cell_edited(self, _renderer, path, new_text, col_idx: int):
        """
        Called by GTK when user edits a chapter cell.
        We validate timestamps before accepting them.
        """
        it = self.store.get_iter(path)
        new_text = (new_text or "").strip()

        # Validate timestamp format
        if col_idx == 0 and new_text:
            try:
                _ = parse_ts_to_ms(new_text)
            except Exception as e:
                self.append_log(f"ERROR: {e}\n")
                return

        self.store.set_value(it, col_idx, new_text)

    def on_add(self):
        """Add a new chapter row after the selected row (or append if none)."""
        model, it = self.tree.get_selection().get_selected()
        if it:
            idx = model.get_path(it).get_indices()[0] + 1
            self.store.insert(idx, ["00:00:00", "New Chapter"])
        else:
            self.store.append(["00:00:00", "New Chapter"])

    def on_remove(self):
        """Remove the selected chapter row."""
        model, it = self.tree.get_selection().get_selected()
        if it:
            model.remove(it)

    def get_rows_validated(self) -> List[Tuple[int, str]]:
        """
        Read chapters from the table and validate required fields.
        Returns list of (start_ms, title).
        """
        rows: List[Tuple[int, str]] = []
        for i, row in enumerate(self.store, start=1):
            ts = (row[0] or "").strip()
            title = (row[1] or "").strip()

            if not ts:
                raise ValueError(f"Row {i}: missing time")
            if not title:
                raise ValueError(f"Row {i}: missing title")

            rows.append((parse_ts_to_ms(ts), title))

        if not rows:
            raise ValueError("No chapters in table")
        return rows

    def on_sort(self):
        """Sort chapter rows by time."""
        try:
            rows = self.get_rows_validated()
        except Exception as e:
            self.append_log(f"ERROR: {e}\n")
            return

        rows.sort(key=lambda x: x[0])
        self.store.clear()
        for ms, title in rows:
            self.store.append([ms_to_ts(ms), title])

        self.append_log("Sorted by time.\n")

    def get_selected_ts_ms(self) -> int:
        """Return the selected chapter's time in ms."""
        model, it = self.tree.get_selection().get_selected()
        if not it:
            raise ValueError("Select a chapter row first.")
        ts = (model.get_value(it, 0) or "").strip()
        if not ts:
            raise ValueError("Selected row has no time.")
        return parse_ts_to_ms(ts)

    def autoload_chapters_if_present(self):
        """
        If current file is .m4b and contains chapters, load them into the table automatically.
        """
        if not self.input_path:
            return
        if file_ext_lower(self.input_path) != ".m4b":
            return

        ch = probe_chapters(self.input_path)
        if not ch:
            self.append_log("No chapters found (auto-load).\n")
            return

        self.store.clear()
        for i, (ms, title) in enumerate(ch, start=1):
            self.store.append([ms_to_ts(ms), title or f"Chapter {i}"])

        self.append_log(f"Auto-loaded {len(ch)} chapters from file.\n")

    # ==========================================================
    # Metadata table logic
    # ==========================================================

    def reset_metadata_table(self):
        """Clear metadata table and seed one example row."""
        self.meta_store.clear()
        self.meta_store.append(["g", "title", ""])

    def on_meta_cell_edited(self, _renderer, path, new_text, col_idx: int):
        """Called when user edits a metadata table cell."""
        it = self.meta_store.get_iter(path)
        self.meta_store.set_value(it, col_idx, (new_text or "").strip())

    def on_meta_add(self, _):
        """Add a metadata row after selection (or append)."""
        model, it = self.meta_tree.get_selection().get_selected()
        if it:
            idx = model.get_path(it).get_indices()[0] + 1
            self.meta_store.insert(idx, ["g", "new_key", "value"])
        else:
            self.meta_store.append(["g", "new_key", "value"])

    def on_meta_remove(self, _):
        """Remove selected metadata row."""
        model, it = self.meta_tree.get_selection().get_selected()
        if it:
            model.remove(it)

    def on_meta_sort(self, _):
        """Sort metadata by (scope, key)."""
        rows = self.get_metadata_rows_validated(allow_empty_value=True)
        rows.sort(key=lambda r: (r[0], r[1].lower()))
        self.meta_store.clear()
        for scope, key, val in rows:
            self.meta_store.append([scope, key, val])
        self.append_log("Metadata sorted.\n")

    def on_meta_clear(self, _):
        """Clear all metadata rows."""
        self.meta_store.clear()
        self.append_log("Metadata cleared.\n")

    def get_metadata_rows_validated(self, allow_empty_value: bool = True) -> List[Tuple[str, str, str]]:
        """
        Return list of (scope, key, value) from metadata table with basic validation.

        Scope rules:
          - 'g' is global/container tags
          - stream scopes must start with 's:' e.g. s:a:0, s:v:0, s:s:0
        """
        out: List[Tuple[str, str, str]] = []
        for i, row in enumerate(self.meta_store, start=1):
            scope = (row[0] or "").strip()
            key = (row[1] or "").strip()
            val = (row[2] or "").strip()

            if not scope:
                raise ValueError(f"Metadata row {i}: missing scope")
            if scope != "g" and not scope.startswith("s:"):
                raise ValueError(f"Metadata row {i}: scope must be 'g' or like 's:a:0' / 's:v:0'")
            if not key:
                raise ValueError(f"Metadata row {i}: missing key")
            if (not allow_empty_value) and (val == ""):
                raise ValueError(f"Metadata row {i}: missing value")

            out.append((scope, key, val))

        return out

    def load_metadata_into_table(self):
        """
        Load all ffprobe-visible embedded metadata into the metadata table.
        - global tags -> scope 'g'
        - per-stream tags -> scope 's:a:N' / 's:v:N' etc.
        """
        self.meta_store.clear()

        if not self.input_path:
            return

        # Global tags
        gtags = probe_format_tags(self.input_path)
        for k, v in sorted(gtags.items(), key=lambda kv: kv[0].lower()):
            self.meta_store.append(["g", k, v])

        # Stream tags
        stream_scoped = probe_stream_tags_with_scopes(self.input_path)
        for scope, tags in stream_scoped:
            for k, v in sorted(tags.items(), key=lambda kv: kv[0].lower()):
                self.meta_store.append([scope, k, v])

        if len(self.meta_store) == 0:
            # If file has no tags, keep table usable by seeding one row
            self.meta_store.append(["g", "title", ""])
            self.append_log("No metadata found; seeded one editable row.\n")
        else:
            self.append_log(f"Loaded metadata rows: {len(self.meta_store)}\n")

    # ==========================================================
    # File selection / auto-load
    # ==========================================================

    def on_choose_file(self, _):
        """
        Choose an input file (.m4b or .mp3).
        This triggers:
        - reset state
        - refresh embedded art preview
        - auto-load metadata + chapters (if present)
        """
        dlg = Gtk.FileChooserDialog(
            title="Select an .m4b or .mp3 file",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
        )
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                        Gtk.STOCK_OPEN, Gtk.ResponseType.OK)

        flt = Gtk.FileFilter()
        flt.set_name("Audiobooks (.m4b, .mp3)")
        flt.add_pattern("*.m4b")
        flt.add_pattern("*.mp3")
        dlg.add_filter(flt)

        if dlg.run() == Gtk.ResponseType.OK:
            self.input_path = dlg.get_filename()
            self.lbl_file.set_text(self.input_path)

            self.reset_for_new_file()
            self.append_log(f"\nSelected: {self.input_path}\n")

            self.refresh_embedded_art_preview()
            self.load_metadata_into_table()
            self.autoload_chapters_if_present()

            self.update_ui_state()

        dlg.destroy()

    # ==========================================================
    # Cover art handling
    # ==========================================================

    def set_pixbuf_preview(self, img_path: str, label: str):
        """
        Load an image file and set it as preview (scaled to fit).
        """
        try:
            pb = GdkPixbuf.Pixbuf.new_from_file(img_path)

            # Keep preview reasonable in size
            max_w, max_h = 260, 260
            w, h = pb.get_width(), pb.get_height()
            scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)

            if scale < 1.0:
                pb = pb.scale_simple(
                    int(w * scale),
                    int(h * scale),
                    GdkPixbuf.InterpType.BILINEAR
                )

            self.art_image.set_from_pixbuf(pb)
            self.art_label.set_text(label)
        except Exception as e:
            self.art_image.clear()
            self.art_label.set_text(f"Could not load art: {e}")

    def refresh_embedded_art_preview(self):
        """
        Refresh art preview:
        - Extract embedded art to a temp PNG if present
        - If user selected a cover image, show that instead
        """
        # Remove old temp preview file to avoid stale display / clutter
        if self.embedded_art_temp and os.path.exists(self.embedded_art_temp):
            try:
                os.remove(self.embedded_art_temp)
            except Exception:
                pass
        self.embedded_art_temp = None

        if not self.input_path:
            self.art_image.clear()
            self.art_label.set_text("No file selected")
            return

        tmp_png = os.path.join("/tmp", f"chaptergui_embedded_art_{os.getpid()}.png")
        if extract_embedded_art_to_png(self.input_path, tmp_png):
            self.embedded_art_temp = tmp_png
            self.set_pixbuf_preview(tmp_png, "Embedded art detected")
        else:
            self.art_image.clear()
            self.art_label.set_text("No embedded art found")

        # If user chose a new art, preview that as the active one
        if self.user_cover_path:
            self.set_pixbuf_preview(self.user_cover_path, f"Chosen art: {os.path.basename(self.user_cover_path)}")

    def on_choose_art(self, _):
        """Select a jpg/png to embed into output."""
        dlg = Gtk.FileChooserDialog(
            title="Choose cover art (jpg/png)",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
        )
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                        Gtk.STOCK_OPEN, Gtk.ResponseType.OK)

        flt = Gtk.FileFilter()
        flt.set_name("Images (.jpg, .png)")
        flt.add_pattern("*.jpg")
        flt.add_pattern("*.jpeg")
        flt.add_pattern("*.png")
        dlg.add_filter(flt)

        if dlg.run() == Gtk.ResponseType.OK:
            self.user_cover_path = dlg.get_filename()
            self.append_log(f"Chosen cover art: {self.user_cover_path}\n")
            self.refresh_embedded_art_preview()
            self.update_ui_state()

        dlg.destroy()

    def on_clear_chosen_art(self, _):
        """Clear chosen art so embedded art (if any) is shown again."""
        self.user_cover_path = None
        self.append_log("Cleared chosen cover art.\n")
        self.refresh_embedded_art_preview()
        self.update_ui_state()

    # ==========================================================
    # Playback (ffplay)
    # ==========================================================

    def start_player(self, start_ms: int):
        """
        Start ffplay at a timestamp.

        Critical: we start ffplay in its own process group (os.setsid) so we can kill the
        whole group reliably. This prevents the “audio keeps playing after GUI closes” issue.
        """
        if not self.input_path:
            return

        start_ms = max(0, int(start_ms))
        ts = ms_to_ts(start_ms)

        # Ensure only one playback instance at a time
        self.stop_player()

        cmd = ["ffplay", "-hide_banner", "-loglevel", "warning", "-nodisp", "-ss", ts, self.input_path]
        self.append_log("Starting ffplay:\n  " + " ".join(shlex.quote(c) for c in cmd) + "\n")

        self.player_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,  # new process group
        )
        self.update_ui_state()

        # Poll to update Play/Stop label when ffplay ends naturally
        def poll_exit():
            if self.player_proc is None:
                return False
            if self.player_proc.poll() is None:
                return True
            self.player_proc = None
            self.append_log("ffplay exited.\n")
            self.update_ui_state()
            return False

        GLib.timeout_add(300, poll_exit)

    def stop_player(self):
        """
        Stop ffplay if running:
        - SIGTERM process group
        - short wait
        - SIGKILL if needed
        """
        if self.player_proc is None:
            return

        try:
            if self.player_proc.poll() is None:
                pgid = os.getpgid(self.player_proc.pid)
                os.killpg(pgid, signal.SIGTERM)

                # Give it a moment to exit cleanly
                for _ in range(10):
                    if self.player_proc.poll() is not None:
                        break
                    time.sleep(0.05)

                # Force kill if still alive
                if self.player_proc.poll() is None:
                    os.killpg(pgid, signal.SIGKILL)
        except Exception as e:
            self.append_log(f"ERROR stopping ffplay: {e}\n")

        self.player_proc = None

    def on_playstop_toggle(self, _):
        """Single Play/Stop toggle button."""
        if self.player_running():
            self.stop_player()
            self.append_log("Stopped.\n")
            self.update_ui_state()
        else:
            self.play_from_selected(0)

    def play_from_selected(self, offset_seconds: int):
        """Play starting at selected chapter timestamp +/- offset_seconds."""
        try:
            sel_ms = self.get_selected_ts_ms()
        except Exception as e:
            self.append_log(f"ERROR: {e}\n")
            return
        self.start_player(sel_ms + offset_seconds * 1000)

    # ==========================================================
    # Silence detection (threaded)
    # ==========================================================

    def on_silence_detect(self, _):
        """
        Start silence scan in a background thread.
        We keep UI responsive and update results with GLib.idle_add().
        """
        if not self.input_path or self.job_running:
            return

        try:
            noise_db = float(self.noise_entry.get_text().strip())
            min_dur = float(self.dur_entry.get_text().strip())
        except Exception:
            self.append_log("ERROR: noise dB and min seconds must be numbers.\n")
            return

        self.append_log(f"Silence scan (noise={noise_db}dB, d={min_dur}s)…\n")
        self.detected_silence_ms = []
        self.update_ui_state()

        def worker():
            ms_list = detect_chapters_from_silence(self.input_path, noise_db, min_dur, mark_offset=1.0)
            GLib.idle_add(self.after_silence_detect, ms_list)

        threading.Thread(target=worker, daemon=True).start()

    def after_silence_detect(self, ms_list: List[int]):
        """
        Receive silence scan results on main thread.
        We log the first 10 and enable Append/Replace/Log buttons.
        """
        self.detected_silence_ms = ms_list
        self.append_log(f"Candidates found: {len(ms_list)}\n")
        for t in ms_list[:10]:
            self.append_log(f"  {ms_to_ts(t)}\n")
        if len(ms_list) > 10:
            self.append_log("  … (use Log ALL)\n")
        self.update_ui_state()

    def on_log_all_detected(self, _):
        """Log all detected candidates."""
        if not self.detected_silence_ms:
            self.append_log("No detected candidates.\n")
            return
        self.append_log(f"ALL detected ({len(self.detected_silence_ms)}):\n")
        for t in self.detected_silence_ms:
            self.append_log(f"  {ms_to_ts(t)}\n")

    def on_append_detected(self, _):
        """
        Append detected candidates as chapters (avoids duplicates).
        Titles are set to 'Auto (silence)' so user can rename.
        """
        if not self.detected_silence_ms:
            self.append_log("No detected candidates to append.\n")
            return

        existing = set()
        for row in self.store:
            try:
                existing.add(parse_ts_to_ms(row[0]))
            except Exception:
                pass

        added = 0
        for ms in self.detected_silence_ms:
            if ms not in existing:
                self.store.append([ms_to_ts(ms), "Auto (silence)"])
                added += 1

        self.append_log(f"Appended {added} chapters. (Tip: Sort)\n")

    def on_replace_with_detected(self, _):
        """Replace chapters table with detected candidates."""
        if not self.detected_silence_ms:
            self.append_log("No detected candidates.\n")
            return

        self.store.clear()
        for ms in self.detected_silence_ms:
            self.store.append([ms_to_ts(ms), "Auto (silence)"])

        self.append_log(f"Replaced with {len(self.detected_silence_ms)} chapters.\n")

    # ==========================================================
    # AAC conversion options
    # ==========================================================

    def on_codec_mode_changed(self, _):
        """
        Toggle enablement between bitrate entry (CBR) and q spinner (VBR).
        """
        is_mp3 = bool(self.input_path) and file_ext_lower(self.input_path) == ".mp3"
        if not is_mp3 or self.job_running:
            self.aac_bitrate.set_sensitive(False)
            self.aac_q.set_sensitive(False)
            return

        is_vbr = self.rb_vbr.get_active()
        self.aac_bitrate.set_sensitive(not is_vbr)
        self.aac_q.set_sensitive(is_vbr)

    def resolve_aac_settings_for_mp3(self, mp3_path: str) -> Tuple[str, str]:
        """
        Return conversion mode and parameter:
          ("vbr", q) or ("cbr", bitrate_str)
        """
        if self.rb_vbr.get_active():
            q = int(self.aac_q.get_value())
            return ("vbr", str(q))

        raw = (self.aac_bitrate.get_text() or "").strip().lower()

        # If auto, try to match the MP3 bitrate
        if raw == "" or raw == "auto":
            bps = detect_mp3_bitrate_bps(mp3_path)
            if bps:
                br = bps_to_aac_bitrate_arg(bps)
                self.ui_log(f"Auto AAC CBR: MP3 ≈ {int(round(bps/1000))}k → using {br}\n")
                return ("cbr", br)
            self.ui_log("Auto AAC CBR: could not detect MP3 bitrate; using 128k\n")
            return ("cbr", "128k")

        bps = parse_bitrate_field_to_bps(raw)
        if bps:
            return ("cbr", bps_to_aac_bitrate_arg(bps))

        # If user typed something unusual, pass it through to ffmpeg
        return ("cbr", raw)

    def convert_mp3_to_m4b(self, mp3_path: str, out_m4b_path: str) -> int:
        """
        Convert MP3 → M4B (AAC audio).
        If MP3 has an attached_pic stream (video), we attempt to preserve it with -c:v copy.
        """
        mode, val = self.resolve_aac_settings_for_mp3(mp3_path)
        hv = has_video_stream(mp3_path)

        cmd = ["ffmpeg", "-y", "-i", mp3_path, "-map", "0:a"]
        if hv:
            cmd += ["-map", "0:v?"]

        cmd += ["-map_metadata", "0", "-c:a", "aac"]
        cmd += (["-q:a", val] if mode == "vbr" else ["-b:a", val])

        if hv:
            cmd += ["-c:v", "copy", "-disposition:v:0", "attached_pic"]

        cmd += ["-sn", "-dn", out_m4b_path]

        self.ui_log("Converting MP3 → M4B:\n  " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        return run_and_stream_output(cmd, self.ui_log)

    # ==========================================================
    # Writing output with chapters + optional cover art
    # ==========================================================

    def write_m4b_with_chapters_and_optional_art(
        self,
        source_path: str,
        meta_path: str,
        out_path: str,
        cover_path: Optional[str]
    ) -> int:
        """
        Write final M4B with chapters and cover art.

        Approach:
        - input 0: source audio (and possibly embedded art)
        - input 1: ffmetadata file (chapters)
        - input 2: optional user cover image

        Then:
        - map audio from source
        - map video either from chosen cover or from source (if exists)
        - copy metadata from source (-map_metadata 0)
        - replace chapters from ffmetadata (-map_chapters 1)
        - copy audio stream (-c:a copy)
        - if embedding new art: encode image and mark attached_pic
          else: copy video stream if present
        """
        embed_new = bool(cover_path) and self.chk_embed_art.get_active()

        cmd = ["ffmpeg", "-y", "-i", source_path, "-i", meta_path]
        if embed_new:
            cmd += ["-i", cover_path]

        cmd += ["-map", "0:a"]
        cmd += (["-map", "2:0"] if embed_new else ["-map", "0:v?"])
        cmd += ["-map_metadata", "0", "-map_chapters", "1", "-sn", "-dn", "-c:a", "copy"]

        if embed_new:
            cmd += ["-c:v", image_codec_for_file(cover_path), "-disposition:v:0", "attached_pic"]
        else:
            cmd += ["-c:v", "copy"]

        cmd += [out_path]

        self.ui_log("Writing output:\n  " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        return run_and_stream_output(cmd, self.ui_log)

    def on_write(self, _):
        """
        Save output to a new .m4b.

        If input is MP3 and checkbox is enabled, converts MP3 → temp M4B first,
        then writes chapters/art into final output.
        """
        if not self.input_path or self.job_running:
            return

        dlg = Gtk.FileChooserDialog(
            title="Save output .m4b as",
            parent=self,
            action=Gtk.FileChooserAction.SAVE,
        )
        dlg.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                        Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
        dlg.set_do_overwrite_confirmation(True)

        base = os.path.splitext(os.path.basename(self.input_path))[0]
        dlg.set_current_name(f"{base}.chapters.m4b")

        out_path = None
        if dlg.run() == Gtk.ResponseType.OK:
            out_path = dlg.get_filename()
        dlg.destroy()

        if not out_path:
            return

        # Validate chapters now so we fail fast (before background job)
        try:
            rows = self.get_rows_validated()
            rows.sort(key=lambda x: x[0])
        except Exception as e:
            self.append_log(f"ERROR: {e}\n")
            return

        ext = file_ext_lower(self.input_path)
        cover_to_embed = self.user_cover_path if self.user_cover_path else None

        def job():
            GLib.idle_add(self._job_begin)
            try:
                source_for_final = self.input_path
                tmp_m4b = out_path + ".tmp_converted.m4b"

                # Optionally convert MP3 to M4B first
                if ext == ".mp3" and self.chk_convert_mp3.get_active():
                    rc = self.convert_mp3_to_m4b(self.input_path, tmp_m4b)
                    if rc != 0:
                        self.ui_log(f"ERROR: MP3→M4B conversion failed (exit code {rc}).\n")
                        return
                    source_for_final = tmp_m4b

                # Build chapter ffmetadata file
                duration_ms = get_duration_ms(source_for_final)
                ffmeta = build_ffmeta(rows, duration_ms)

                meta_path = out_path + ".ffmeta"
                with open(meta_path, "w", encoding="utf-8") as f:
                    f.write(ffmeta)

                # Write final output
                rc2 = self.write_m4b_with_chapters_and_optional_art(
                    source_for_final, meta_path, out_path, cover_to_embed
                )

                self.ui_log(f"\nDone (exit code {rc2}). Output:\n  {out_path}\n")

                # Cleanup temporary conversion
                if os.path.exists(tmp_m4b):
                    try:
                        os.remove(tmp_m4b)
                        self.ui_log(f"Cleaned up temp: {tmp_m4b}\n")
                    except Exception as e:
                        self.ui_log(f"Warning: could not remove temp file: {e}\n")

            except Exception as e:
                self.ui_log(f"ERROR: {e}\n")
            finally:
                GLib.idle_add(self._job_end)

        threading.Thread(target=job, daemon=True).start()

    def _job_begin(self):
        """Mark busy and disable relevant UI."""
        self.job_running = True
        self.update_ui_state()

    def _job_end(self):
        """Clear busy and re-enable UI."""
        self.job_running = False
        self.update_ui_state()

    # ==========================================================
    # Apply metadata (generic table)
    # ==========================================================

    def apply_metadata_ffmpeg(self, in_path: str, out_path: str) -> int:
        """
        Apply metadata from meta_store.

        Strategy:
        - stream copy everything (-c copy)
        - clear global metadata (-map_metadata -1)
        - set global tags:   -metadata key=value
        - set stream tags:   -metadata:<scope> key=value   (scope like s:a:0)
        """
        rows = self.get_metadata_rows_validated(allow_empty_value=True)

        # Partition by global vs stream
        global_rows = [(k, v) for (scope, k, v) in rows if scope == "g"]
        stream_rows = [(scope, k, v) for (scope, k, v) in rows if scope != "g"]

        cmd = ["ffmpeg", "-y", "-i", in_path, "-map", "0", "-c", "copy"]

        # Clear global tags, then write global tags explicitly
        cmd += ["-map_metadata", "-1"]
        for k, v in global_rows:
            cmd += ["-metadata", f"{k}={v}"]

        # Write stream-scoped tags
        for scope, k, v in stream_rows:
            cmd += [f"-metadata:{scope}", f"{k}={v}"]

        cmd += ["-sn", "-dn", out_path]

        self.ui_log("Applying metadata:\n  " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        return run_and_stream_output(cmd, self.ui_log)

    def on_apply_metadata(self, _):
        """
        Apply metadata changes to either:
        - overwrite original file (keeping .bak)
        - or write a .metadata copy and switch the app to it
        """
        if not self.input_path or self.job_running:
            return

        overwrite = self.chk_overwrite_meta.get_active()
        in_path = self.input_path
        base_dir = os.path.dirname(in_path)
        base_name = os.path.basename(in_path)
        ext = file_ext_lower(in_path)

        # Write to a hidden temp file in the same directory first
        tmp_out = os.path.join(base_dir, f".{base_name}.tmp_meta{ext}")

        def job():
            GLib.idle_add(self._job_begin)
            try:
                rc = self.apply_metadata_ffmpeg(in_path, tmp_out)
                if rc != 0:
                    self.ui_log(f"ERROR: metadata apply failed (exit code {rc}).\n")
                    return

                if overwrite:
                    backup = os.path.join(base_dir, f"{base_name}.bak")
                    try:
                        if os.path.exists(backup):
                            os.remove(backup)
                        os.rename(in_path, backup)
                        os.rename(tmp_out, in_path)
                        self.ui_log(f"Overwrote original. Backup kept at:\n  {backup}\n")
                    except Exception as e:
                        self.ui_log(f"ERROR: could not overwrite original: {e}\n")
                        return
                else:
                    out_path = os.path.join(base_dir, f"{os.path.splitext(base_name)[0]}.metadata{ext}")
                    try:
                        if os.path.exists(out_path):
                            os.remove(out_path)
                        os.rename(tmp_out, out_path)
                        self.input_path = out_path
                        GLib.idle_add(self.lbl_file.set_text, self.input_path)
                        self.ui_log(f"Saved metadata-updated copy:\n  {out_path}\nNow working on that file.\n")
                    except Exception as e:
                        self.ui_log(f"ERROR: could not finalize metadata copy: {e}\n")
                        return

                # Reload views because file metadata/art may have changed
                GLib.idle_add(self.load_metadata_into_table)
                GLib.idle_add(self.refresh_embedded_art_preview)
                GLib.idle_add(self.update_ui_state)

            finally:
                # Clean temp if still present
                try:
                    if os.path.exists(tmp_out):
                        os.remove(tmp_out)
                except Exception:
                    pass
                GLib.idle_add(self._job_end)

        threading.Thread(target=job, daemon=True).start()

    # ==========================================================
    # Window close cleanup
    # ==========================================================

    def on_window_delete(self, *_args):
        """
        Ensure no background audio keeps playing and temp preview files are removed.
        """
        self.stop_player()

        if self.embedded_art_temp and os.path.exists(self.embedded_art_temp):
            try:
                os.remove(self.embedded_art_temp)
            except Exception:
                pass

        return False


# ==========================================================
# GTK entry point
# ==========================================================

def main():
    win = ChapterGUI()
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
