import io

import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image

LOW_CUT = 100
HIGH_CUT = 1800


def apply_spectrogram_colormap(normalized: np.ndarray) -> np.ndarray:
    """Map a normalized STFT matrix for the standalone web spectrogram."""
    values = np.clip(normalized, 0.0, 1.0)
    return colormaps["magma_r"](values, bytes=True)[..., :3]


def apply_model_spectrogram_colormap(normalized: np.ndarray) -> np.ndarray:
    """Map a normalized STFT matrix to the palette used by Grad-CAM/overlay."""
    values = np.clip(normalized, 0.0, 1.0)
    return colormaps["magma"](values, bytes=True)[..., :3]


def apply_gradcam_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Map Grad-CAM contribution values to the same jet palette as the notebook."""
    values = np.clip(heatmap, 0.0, 1.0)
    return colormaps["jet"](values, bytes=True)[..., :3]


def render_axis_panel(
    image: np.ndarray,
    *,
    show_contribution_scale: bool = False,
    show_waveform_legend: bool = False,
) -> Image.Image:
    """Render a web-facing panel with complete frequency and time-frame axes."""
    height, width = image.shape[:2]
    figure = Figure(figsize=(5.2, 4.4), dpi=150, facecolor="white")
    axis = figure.subplots()
    axis.imshow(image)

    frequency_ticks = np.linspace(LOW_CUT, HIGH_CUT, 6)
    pixel_ticks = (HIGH_CUT - frequency_ticks) / (HIGH_CUT - LOW_CUT) * (height - 1)
    axis.set_yticks(pixel_ticks)
    axis.set_yticklabels([f"{frequency:.0f} Hz" for frequency in frequency_ticks])
    time_ticks = np.arange(0, width, 50)
    axis.set_xticks(time_ticks)
    axis.set_xticklabels([str(int(frame)) for frame in time_ticks])
    axis.set_xlabel("Time Frame")
    axis.set_ylabel("Frequency (Hz)")

    if show_contribution_scale:
        scale = ScalarMappable(cmap="jet", norm=Normalize(0, 1))
        scale.set_array([])
        colorbar = figure.colorbar(scale, ax=axis, fraction=0.046, pad=0.04)
        colorbar.set_label("Contribution to Prediction")
        colorbar.set_ticks([0, 0.5, 1])
        colorbar.set_ticklabels(["Low", "Medium", "High"])

    if show_waveform_legend:
        axis.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="#FFE600",
                    linewidth=2,
                    label="Raw audio signal",
                )
            ],
            loc="lower right",
            framealpha=0.7,
        )

    figure.tight_layout(pad=0.8)
    with io.BytesIO() as output:
        figure.savefig(output, format="png", facecolor="white")
        output.seek(0)
        with Image.open(output) as rendered:
            result = rendered.convert("RGB").copy()

    figure.clear()
    return result
