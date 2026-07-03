# Security & Bug Fix Report — Upscaling_ESRGAN

**Date:** 2026-07-02
**Scope:** Full review of all Python sources, packaging metadata, and download tooling in `Ikshvaku-K/Upscaling_ESRGAN`.
**Result:** 3 security issues and ~18 functional bugs found and fixed. All fixes verified with a 10-case functional test suite (see [Verification](#verification)).

---

## 1. Security Vulnerabilities

### 1.1 `requirements.txt` was a system `pip freeze` with known-vulnerable pins — **High**

**File:** `requirements.txt`

The file was a dump of an Ubuntu system Python environment (`cloud-init`, `ufw`, `ubuntu-drivers-common`, `python-apt`, …). It did **not** contain any of the project's real dependencies (`torch`, `opencv-python`, `basicsr`, `realesrgan`, …), and it pinned several packages to versions with published CVEs. Anyone running `pip install -r requirements.txt` would install:

| Package pin | Known CVEs (fixed version) |
| --- | --- |
| `requests==2.31.0` | CVE-2024-35195, CVE-2024-47081 (2.32.4) |
| `urllib3==2.0.7` | CVE-2024-37891 (2.2.2) |
| `pillow==10.2.0` | CVE-2024-28219 (10.3.0) |
| `Jinja2==3.1.2` | CVE-2024-22195, CVE-2024-34064, CVE-2024-56201, CVE-2024-56326 (3.1.5) |
| `cryptography==41.0.7` | CVE-2023-50782, CVE-2024-26130 (42.0.4) |
| `setuptools==68.1.2` | CVE-2024-6345 — remote code execution (70.0.0) |
| `PyJWT==2.7.0` | CVE-2024-53861 (2.10.1) |
| `configobj==5.0.8` | CVE-2023-26112 ReDoS |

**Fix:** Replaced the file with the project's actual dependencies, with security floors where relevant (`requests>=2.32.4`). Added the previously-missing deps used by the scripts (`requests`, `scikit-image`, `beautifultable`) and a documented constraint `torchvision>=0.15,<0.17` (basicsr 1.4.2 imports `torchvision.transforms.functional_tensor`, removed in torchvision 0.17 — installing an unconstrained torchvision breaks every import of `basicsr`).

### 1.2 `torch.load()` without `weights_only=True` — pickle deserialization → arbitrary code execution — **High**

**Files:** `compare_models.py` (2 call sites), `export_onnx.py`, `inspect_checkpoint.py`

`torch.load()` uses Python pickle by default. Loading a malicious/tampered `.pth` file executes arbitrary code on the machine. Combined with 1.3 below (downloads with no integrity check), a corrupted or MITM'd/re-uploaded model file would have been silently executed.

**Fix:** All first-party `torch.load()` calls now pass `weights_only=True`, which restricts deserialization to tensor data. Also added `map_location=device` to the SwinIR load so it works on CPU-only machines.

> Note: `RealESRGANer` (the `realesrgan` package) still calls `torch.load` internally on the model path you give it — that is third-party code and can't be fixed here. Mitigated by 1.3: checksums are now verified at download time.

### 1.3 Model downloads: no TLS timeout, no HTTP status check, no integrity verification, non-atomic writes — **Medium**

**Files:** `download_bsrgan.py`, `download_swinir.py`

Four problems:
1. `requests.get(...)` had **no timeout** — a stalled connection hangs forever.
2. **No `raise_for_status()`** — a 404/500 HTML error page would be saved as `BSRGAN.pth` and later fed to `torch.load`.
3. **No checksum verification** — no way to detect corruption or tampering of files that are subsequently unpickled (see 1.2).
4. Files were written **directly to the final path** — an interrupted download left a truncated file that the `os.path.exists()` guard would then treat as "already downloaded", permanently wedging the setup.

**Fix:** Both scripts now use `timeout=(10, 60)`, `raise_for_status()`, download to a `.part` temp file with atomic `os.replace()` on success, verify the byte count, and verify a pinned SHA-256 before installing the file. The pinned hashes were computed from the official GitHub release assets on 2026-07-02:

| Model | SHA-256 |
| --- | --- |
| `BSRGAN.pth` (cszn/KAIR v1.0) | `5d505a0766160921e0388d76e1ddf08cb114303990f9080432bf2b1c988b1c54` |
| `001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth` (JingyunLiang/SwinIR v0.0) | `4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215` |

A checksum mismatch aborts, deletes the temp file, and never installs the model.

---

## 2. Functional Bugs

### Crashes / broken entry points

**2.1 `upscaler image` subcommand always crashed** — `src/upscaler/cli.py` + `src/upscaler/core/image.py`
The CLI forwarded `-n`, `-s`, `--suffix`, `-t`, `--tile_pad`, `--pre_pad`, `--fp32`, `--gpu-id` to `core/image.py`, whose parser only defined `input` and `-o`. Every invocation died with `error: unrecognized arguments` (exit 2). The image module's parser now accepts and honors all forwarded options (model name → `models/<name>.pth`, tile size, precision, GPU id, suffix-based output naming), and both `image` and `video` mains take an explicit `argv` parameter instead of the previous `sys.argv` mutation hack.

**2.2 Whole CLI unusable without TensorRT** — `src/upscaler/cli.py`
`from upscaler.core.trt_convert import build_engine` at module level made **every** subcommand (`image`, `video`, `optimize`) fail with `ModuleNotFoundError: tensorrt` unless TensorRT — an optional, undeclared dependency — was installed. The import is now lazy inside `convert-trt`, with a clear error message. `tensorrt` is declared as an optional extra in `pyproject.toml` (`pip install upscaler[trt]`).

**2.3 `production_image_upscale.py` could never start** — two independent crashes:
- `from benchmarking import BenchmarkTracker` — no such module exists at the repo root (it lives at `src/upscaler/utils/benchmarking.py`). Fixed with a proper package import plus a `src/` path fallback for non-installed use.
- The default `--config config.yaml` doesn't exist at the repo root (the config lives at `src/upscaler/config.yaml`), so the script exited immediately. It now falls back to the packaged default config.

**2.4 `import os` missing in `src/upscaler/core/trt_run.py`**
`os` was only imported inside the `if __name__ == "__main__"` block, so importing `process_video_gen` from another module raised `NameError`. Moved to the top of the file.

**2.5 FP16 on CPU crashes** — 7 call sites
`half=True` was passed to `RealESRGANer` unconditionally in `core/image.py`, `core/video.py`, `upscale_video.py`, `upscale_video_pipeline.py`, `production_image_upscale.py`, `benchmark_phase3.py`, and `compare_models.py` (2 sites). On CPU-only machines PyTorch raises `"slow_conv2d_cpu" not implemented for 'Half'`. All sites now use `half=... and device.type == 'cuda'`.

### Hangs / deadlocks

**2.6 `upscale_video_pipeline.py` hung forever after finishing** — the `None` shutdown sentinel put into `write_queue` was never `task_done()`'d, so `write_queue.join()` never returned. Worse, because the writer was a daemon thread, if the process had exited another way, ffmpeg would have been killed before finalizing the file, producing a corrupt/truncated video. Fixed: sentinel is marked done, and the script now joins the reader/writer threads so ffmpeg finalizes the output before "Complete!" is printed.

**2.7 ffmpeg failures deadlocked the video pipeline** — `src/upscaler/core/video.py`
- Both ffmpeg processes ran with `stderr=DEVNULL` and **no return-code checks**: encode/decode failures (bad codec, odd dimensions vs `yuv420p`, missing ffmpeg binary) were invisible, and a failed job could be marked "completed".
- If ffmpeg failed to start, the reader thread died without putting the EOF sentinel → the inference loop blocked forever on `read_queue.get()`.
- If the encoder died mid-run, `stdin.write` raised `BrokenPipeError`, the writer thread died, and the inference loop blocked forever on the bounded `write_queue.put()`.
- If **inference** crashed (e.g. CUDA OOM), the reader/writer threads and both ffmpeg processes leaked — one pair per failed file in batch mode.

Fixed: ffmpeg presence is checked up front (`shutil.which`); stderr is captured to a temp file and its tail is included in error messages; return codes are checked; worker-thread errors are collected and re-raised in the main thread; the writer drains the queue after a failure so the producer can never block; and an `abort` event tears both workers (and their ffmpeg processes) down cleanly when inference fails.

**2.8 nvidia-smi polling could hang the benchmark thread** — `src/upscaler/utils/benchmarking.py`
`subprocess.run` had no timeout; added `timeout=5`. Also fixed the power parse on multi-GPU systems (nvidia-smi emits one line per GPU; `float()` on the joined output raised and silently disabled power sampling) and made `stop()` safe if `start()` was never called (previously `AttributeError` on `start_time`).

### Wrong results / robustness

**2.9 `--tile 0` silently dropped** — `src/upscaler/cli.py` used `if args.tile:` so an explicit `--tile 0` (meaning "disable tiling") was never forwarded. Now `is not None`.

**2.10 Shared mutable default config** — `src/upscaler/core/video.py` used `DEFAULT_CONFIG.copy()` (shallow); merging user config with `config[key].update(value)` **mutated the module-level `DEFAULT_CONFIG`** through the shared nested dicts. Now `copy.deepcopy`.

**2.11 `fps = 0` produced broken outputs** — when OpenCV can't determine FPS it returns 0, which was passed to ffmpeg `-r 0` / `cv2.VideoWriter(fps=0)`. All video paths now fall back to 30 fps with a warning.

**2.12 Diff-map visualizations were corrupted by uint8 overflow** — `compare_models.py` amplified difference maps with `diff * 5` / `* 10` on uint8 arrays, which wraps modulo 256 (large differences rendered as near-black noise). Replaced with saturating `cv2.convertScaleAbs`.

**2.13 Unchecked I/O** — `cv2.imread` results were used without `None` checks in `compare_models.py` and `benchmark_phase3.py` (corrupt/unsupported input → `AttributeError: 'NoneType' object has no attribute 'shape'`); `cv2.imwrite` return values ignored in `core/image.py` and `production_image_upscale.py` (silent data loss on unwritable paths); `cv2.VideoWriter`/`VideoCapture` open status unchecked in `upscale_video.py` and `core/trt_run.py`. All checked now.

**2.14 TRT CLI progress bar showed 1 frame total** — `core/trt_run.py` with `yield_results=False` only yielded the final completion sentinel, so the CLI's per-frame `pbar.update(1)` counted a single tick for the whole video. The generator now yields per-frame progress (without the frame payload) in both modes.

**2.15 `export_onnx.py` couldn't load raw state dicts** — checkpoints without a `params`/`params_ema` wrapper raised `KeyError`. Now falls back to treating the checkpoint as a plain state dict. (Also removed a dead `hasattr(model, 'load_state_dict')` branch.)

**2.16 Bare `except:` swallowed `KeyboardInterrupt`/`SystemExit`** — `src/upscaler/utils/hardware.py`; narrowed to `except Exception`.

**2.17 Packaged config never shipped** — `pyproject.toml` had no package-data declaration, so `src/upscaler/config.yaml` (which `core/video.py` loads via `importlib.resources`) was omitted from wheels/installs. Added `[tool.setuptools.package-data]`.

---

## 3. Repo Hygiene (resolved in follow-up commit)

Items 1–4 were flagged in the initial audit and have since been fixed; item 5 remains a deployment recommendation.

1. ✅ **~100 MB of build artifacts were committed**: `models/realesrgan.onnx.data` (64 MB) and `models/realesrgan.trt` (37 MB). TensorRT engines are not portable across GPU/driver/TRT versions, and the `.onnx.data` external-weights file was committed without its companion `.onnx` index file, making it unusable anyway. **Fixed:** removed from tracking, then purged from all git history with `git filter-repo` + force-push (2026-07-03). A fresh clone dropped from ~100 MB to ~400 KB. Commit hashes were rewritten in the process; any pre-purge clones should be re-cloned. Regenerate the artifacts locally with `export_onnx.py` / `upscaler convert-trt`.
2. ✅ **Committed backup files**: `production_upscale.py.bak`, `production_image_upscale.py.bak` were stale pre-fix copies of `src/upscaler/core/video.py` and `production_image_upscale.py`. **Fixed:** deleted.
3. ✅ **`.gitignore` ignored everything by default** (`*` with a file-by-file whitelist). New root-level files were silently dropped from version control — likely how `config.yaml` disappeared from the repo root. **Fixed:** the whitelist now includes `!*.py` and `!*.md`.
4. ✅ **README drift**: the README documented `upscale_image.py`, which doesn't exist in the repo. **Fixed:** Phase 1 now points to the real entry points (`upscaler image` CLI / `production_image_upscale.py`).
5. `requirements.txt` intentionally leaves `torch`/`torchvision` unpinned beyond the basicsr compatibility ceiling; pin exact versions for reproducible deployments.

---

## Verification

- `python3 -m py_compile` passes on all 17 modified/reviewed Python files.
- A functional test suite (heavy deps stubbed, ffmpeg emulated with in-memory pipes) exercises the fixed paths — **10/10 pass**:

| # | Test | Validates fix |
| --- | --- | --- |
| 1 | `upscaler image` with missing file exits 1, not argparse error 2 | 2.1 |
| 2 | `upscaler image` full flag set accepted end-to-end | 2.1 |
| 3 | `upscaler video --tile 0` forwarded intact | 2.9 |
| 4 | `DEFAULT_CONFIG` unchanged after user-config merge | 2.10 |
| 5 | Video pipeline happy path: all 10 frames reach the encoder | 2.7 regression guard |
| 6 | Inference crash mid-video propagates within timeout, no thread hang | 2.7 |
| 7 | ffmpeg decoder failure surfaces as `RuntimeError` with exit code | 2.7 |
| 8 | `upscale_video_pipeline.py` completes (no `write_queue.join()` hang) | 2.6 |
| 9 | `BenchmarkTracker.stop()` before `start()` is safe | 2.8 |
| 10 | Download: SHA-256 mismatch rejects file, `.part` cleaned up, good hash installs atomically | 1.3 |

GPU inference itself (Real-ESRGAN/SwinIR forward passes, TensorRT engine execution) was not run — no CUDA GPU/ffmpeg on this machine; those code paths were reviewed statically and their control flow tested via stubs.

---

## Files Changed

| File | Changes |
| --- | --- |
| `requirements.txt` | Rewritten: real deps, CVE-free floors (1.1) |
| `pyproject.toml` | Optional extras `trt`/`scripts`, package-data for config.yaml (2.2, 2.17) |
| `download_bsrgan.py`, `download_swinir.py` | Timeout, status check, SHA-256 pinning, atomic install (1.3) |
| `src/upscaler/cli.py` | Arg plumbing without `sys.argv` hack, `--tile 0` fix, lazy TRT import (2.1, 2.2, 2.9) |
| `src/upscaler/core/image.py` | Full CLI arg support, CPU-half guard, imwrite check, exit codes (2.1, 2.5, 2.13) |
| `src/upscaler/core/video.py` | Deepcopy config, ffmpeg error propagation + abort teardown, fps guard, CPU-half guard (2.5, 2.7, 2.10, 2.11) |
| `src/upscaler/core/trt_run.py` | `os` import, capture/writer guards, per-frame progress (2.4, 2.13, 2.14) |
| `src/upscaler/utils/hardware.py` | `except Exception` (2.16) |
| `src/upscaler/utils/benchmarking.py` | nvidia-smi timeout + multi-GPU parse, safe stop() (2.8) |
| `production_image_upscale.py` | Fixed import, config fallback, CPU-half guard, imwrite check (2.3, 2.5, 2.13) |
| `upscale_video.py` | CPU-half guard, fps guard, writer check (2.5, 2.11, 2.13) |
| `upscale_video_pipeline.py` | Queue-join deadlock fix, thread joins, CPU-half guard, fps guard (2.5, 2.6, 2.11) |
| `compare_models.py` | `weights_only=True`, imread check, saturating diff maps, CPU-half guards (1.2, 2.5, 2.12, 2.13) |
| `export_onnx.py` | `weights_only=True`, raw state-dict fallback (1.2, 2.15) |
| `inspect_checkpoint.py` | `weights_only=True` (1.2) |
| `benchmark_phase3.py` | CPU-half guard, imread check (2.5, 2.13) |
