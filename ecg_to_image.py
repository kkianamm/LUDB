"""
Turn a 12-lead PTB-XL waveform into an ECG-paper-style image that BiomedCLIP
(a vision-language model trained on biomedical *figures*) can ingest.

Why images?  BiomedCLIP is a CLIP-style image<->text model.  PTB-XL ships raw
signals (WFDB).  We therefore rasterise each recording into a PNG that visually
resembles the ECG figures found in biomedical papers (red grid, stacked leads).

Public entry points:
    load_signal(data_dir, filename)      -> (np.ndarray[N, 12], fields dict)
    render_ecg(signal, fs, ...)          -> PIL.Image
    render_to_file(signal, fs, out_path) -> saves PNG
"""
import io
import os

import numpy as np
import wfdb
import matplotlib
matplotlib.use("Agg")  # headless / server-safe backend
import matplotlib.pyplot as plt
from PIL import Image

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


def load_signal(data_dir, filename):
    """Read one WFDB record.

    `filename` is the value from ptbxl_database.csv, e.g.
    'records100/00000/00001_lr' (no extension). Returns (signal, fields)
    where signal has shape (n_samples, 12) in millivolts.
    """
    path = os.path.join(data_dir, filename)
    signal, fields = wfdb.rdsamp(path)
    return signal.astype(np.float32), fields


def render_ecg(signal, fs=100, style="grid", dpi=100, line_width=0.7):
    """Render a 12-lead ECG to a PIL image.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples, 12), millivolts
    fs     : sampling frequency in Hz (100 or 500)
    style  : "grid"  -> ECG-paper look (pink background, red grid)
             "plain" -> white background, no grid
    Returns
    -------
    PIL.Image (RGB)
    """
    n_samples, n_leads = signal.shape
    t = np.arange(n_samples) / fs
    duration = n_samples / fs

    # One row per lead, stacked; each row shows the full 10 s so no data is lost.
    fig, axes = plt.subplots(
        n_leads, 1, figsize=(6, 8), dpi=dpi, sharex=True
    )
    fig.subplots_adjust(hspace=0.0, left=0.06, right=0.99, top=0.99, bottom=0.04)

    for i, ax in enumerate(axes):
        ax.plot(t, signal[:, i], color="black", linewidth=line_width)

        if style == "grid":
            ax.set_facecolor("#fff5f5")
            # Fine grid: 0.04 s and 0.1 mV; coarse grid: 0.2 s and 0.5 mV
            ax.set_xticks(np.arange(0, duration + 0.01, 0.2), minor=True)
            ax.set_xticks(np.arange(0, duration + 0.01, 1.0), minor=False)
            ax.grid(which="minor", color="#f4b6b6", linewidth=0.3)
            ax.grid(which="major", color="#e07a7a", linewidth=0.6)

        ax.set_ylabel(LEAD_NAMES[i], rotation=0, labelpad=14,
                      va="center", fontsize=7)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[-1].set_xlabel("time (s)", fontsize=7)

    # Rasterise the figure into a PIL image (no temp file needed).
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_to_file(signal, fs, out_path, **kwargs):
    img = render_ecg(signal, fs=fs, **kwargs)
    img.save(out_path)
    return out_path


if __name__ == "__main__":
    # Smoke test with a synthetic sine "ECG" (no dataset needed).
    fs = 100
    t = np.arange(fs * 10) / fs
    fake = np.stack([np.sin(2 * np.pi * 1.2 * t + k) for k in range(12)], axis=1)
    img = render_ecg(fake, fs=fs)
    print("Rendered image size:", img.size)
    render_to_file(fake, fs, "example_ecg.png")
    print("Saved example_ecg.png")
