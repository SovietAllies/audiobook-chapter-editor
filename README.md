# Audiobook Editor

A **GTK3** desktop app for editing **chapters**, **cover art**, and **metadata** in audiobook files (`.m4b`). It can also convert `.mp3` to `.m4b` and then write chapters.

GitHub: https://github.com/SovietAllies/audiobook-chapter-editor

## Features

<img width="1403" height="909" alt="image" src="https://github.com/user-attachments/assets/f20d2b88-c728-41bb-b0d9-8523ad8dc2c6" />

### Chapters
- Auto-load existing chapters from `.m4b`
- Add/edit/sort chapters using normal timestamps (`HH:MM:SS` or `HH:MM:SS.mmm`)
- Preview audio using **ffplay** from a selected timestamp (with ± offsets)
- Find chapter candidates via **silence detection**

### Cover art
- Auto-detect and preview embedded cover art (e.g., `attached_pic`)
- Embed a new JPG/PNG cover image into the output file

### Metadata (generic key/value editor)
- Loads all metadata visible via `ffprobe`
- Edit **global** (`g`) and **stream-scoped** (`s:a:0`, `s:v:0`, etc.) tags
- Apply changes safely (write a copy or overwrite with a `.bak` backup)

### MP3 → M4B conversion
- Optional conversion workflow when input is MP3
- AAC **VBR** or **CBR** settings (with an “auto” bitrate option)

## Requirements

- `ffmpeg`, `ffprobe`, `ffplay` available on your PATH
- Python 3
- GTK3 Python bindings (PyGObject)

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y ffmpeg python3-gi gir1.2-gtk-3.0
```

## Run

```bash
python3 Audiobook_Editor.py
```

## Settings

Audiobook Editor remembers:
- the last folder you opened files from
- the last folder you saved files to

It stores these in:

- Linux: `~/.config/ChapterGUI/settings.json`

(Directory name is kept for backward compatibility with earlier builds.)

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

- You may use, modify, and share this program.
- If you distribute a modified version (or run it as a network service), you must provide the corresponding source code.

See `LICENSE`.

> Note: AGPL is a copyleft license. It does not prohibit charging money, but it does require source-code availability for distributed/hosted modifications.

## Credits

This project was developed with the help of **ChatGPT (GPT-5.2)**.



