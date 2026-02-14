# Contributing to Audiobook Editor

Thanks for considering contributing!

## License of contributions

By submitting a pull request, you agree that your contributions will be licensed under the **AGPL-3.0**.

## Development notes

- Prefer small, focused PRs.
- Keep UI responsive: run long `ffmpeg` operations in worker threads and use `GLib.idle_add()` to update GTK widgets.
- Be careful managing `ffplay` processes (start in a new process group so Stop/Exit reliably ends playback).

## Style

- Keep code readable.
- Add comments where `ffmpeg`/`ffprobe` flags or stream mapping is non-obvious.

## Testing checklist

- Load an `.m4b` with chapters and confirm they auto-populate.
- Test `Play`/`Stop` and ensure playback stops when closing the app.
- Test silence detection and adding/replacing chapters.
- Test writing a new `.m4b` and verify chapters appear in a player.
- Test embedded cover art extraction + embedding a new image.
- Test Metadata tab load/edit/apply.

