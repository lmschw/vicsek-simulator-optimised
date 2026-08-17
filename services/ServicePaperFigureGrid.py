import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

"""
Generic multi-panel ("grid") figure renderer for the paper figures (see paper_figures_specs.py and
run_paper_figures.py). Handles: a shared per-figure legend, column headers (top row only), row
headers (left column only), a shared y-axis label repeated on the left column of every row and a
shared x-axis label repeated on the bottom row of every column (per the user's spec), and
paper-appropriate font sizing. Saves every figure as pdf, svg and png.

Colours are the first two/five slots of the validated colour-blind-safe categorical palette (see
the dataviz skill's references/palette.md) - slot 1 (blue) is always "ordered start", slot 2
(orange) always "disordered start", so colour usage is consistent across every figure in the paper.
"""

CATEGORICAL_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ORDERED_COLOUR = CATEGORICAL_COLOURS[0]
DISORDERED_COLOUR = CATEGORICAL_COLOURS[1]
EVENT_SHADING_COLOUR = "#1baf7a"

STANDARD_LEGEND = [("ordered start", ORDERED_COLOUR, "-"), ("disordered start", DISORDERED_COLOUR, "-")]

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "svg.fonttype": "none",  # keep text as text (not paths) in the SVG output
})


def seriesFor(t, mean, std, label, colour, linestyle="-"):
    return {"t": t, "mean": mean, "std": std, "label": label, "colour": colour, "linestyle": linestyle}


def orderedDisorderedSeries(orderedResult, disorderedResult):
    """
    orderedResult/disorderedResult: (t, mean, std) tuples as returned by
    ServicePaperFigureRunner.evaluateCell, or None if no data. Returns the list of series dicts
    for one grid cell using the standard ordered/disordered colour convention.
    """
    series = []
    if orderedResult is not None:
        t, mean, std = orderedResult
        series.append(seriesFor(t, mean, std, "ordered start", ORDERED_COLOUR))
    if disorderedResult is not None:
        t, mean, std = disorderedResult
        series.append(seriesFor(t, mean, std, "disordered start", DISORDERED_COLOUR))
    return series


def _plotCellSeries(ax, series, backgroundSpan=None):
    maxT = None
    for s in series:
        ax.plot(s["t"], s["mean"], color=s["colour"], linestyle=s.get("linestyle", "-"), linewidth=1.2)
        if s.get("std") is not None:
            ax.fill_between(s["t"], s["mean"] - s["std"], s["mean"] + s["std"], color=s["colour"], alpha=0.2, linewidth=0)
        maxT = s["t"][-1] if maxT is None else max(maxT, s["t"][-1])
    if backgroundSpan is not None:
        ax.axvspan(backgroundSpan[0], backgroundSpan[1], color=EVENT_SHADING_COLOUR, alpha=0.12, linewidth=0)
    # no autoscale margin on the x-axis - the data should run flush to the right edge of the axes
    # rather than leaving a gap past the last timestep (matplotlib's default ~5% margin).
    ax.set_xlim(0, maxT)


def _legendHandles(legendEntries):
    handles = [Line2D([0], [0], color=colour, linestyle=linestyle, linewidth=1.5) for (_, colour, linestyle) in legendEntries]
    labels = [label for (label, _, _) in legendEntries]
    return handles, labels


def _save(fig, outputPathBase):
    # bbox_inches="tight" crops the saved figure to exactly the drawn content (axes, headers, and
    # the legend, which is placed just outside the axes area below) - rather than reserving a fixed
    # fraction of the figure for the legend via tight_layout's rect, which under- or over-shoots
    # depending on how many rows the grid has, leaving arbitrary blank space.
    os.makedirs(os.path.dirname(outputPathBase), exist_ok=True)
    fig.savefig(f"{outputPathBase}.pdf", bbox_inches="tight")
    fig.savefig(f"{outputPathBase}.svg", bbox_inches="tight")
    fig.savefig(f"{outputPathBase}.png", dpi=300, bbox_inches="tight")


def renderGrid(cells, nRows, nCols, rowHeaders, colHeaders, yLabel, xLabel, legendEntries,
               outputPathBase, ylim=None, backgroundSpan=None, figSizePerCell=(2.3, 1.8), cellTitles=None):
    """
    cells: {(row, col): [series dict, ...]} as produced by orderedDisorderedSeries() or seriesFor().
    rowHeaders / colHeaders: list[str] (length nRows / nCols) or None to omit.
    legendEntries: list of (label, colour, linestyle) tuples for the one shared legend.
    backgroundSpan: a single (start, end) tuple applied to every cell, a {(row, col): (start, end)}
    dict for per-cell spans (e.g. event duration varies per row), or None.
    cellTitles: {(row, col): str} to give every individual cell its own title (e.g. fig_2's per-cell
    mechanism names), instead of/in addition to the row/column headers.
    """
    # extra vertical room between rows only needed when they carry their own top title (cellTitles) -
    # row headers are placed to the side (see below), not above, so they don't need it.
    hspace = 0.55 if cellTitles is not None else 0.3
    fig, axes = plt.subplots(nRows, nCols, figsize=(nCols * figSizePerCell[0], nRows * figSizePerCell[1]),
                              sharex=True, sharey=True, squeeze=False, gridspec_kw={"hspace": hspace, "wspace": 0.15})
    for r in range(nRows):
        for c in range(nCols):
            ax = axes[r][c]
            cellSpan = backgroundSpan.get((r, c)) if isinstance(backgroundSpan, dict) else backgroundSpan
            _plotCellSeries(ax, cells.get((r, c), []), backgroundSpan=cellSpan)
            if ylim is not None:
                ax.set_ylim(ylim)
            if cellTitles is not None and (r, c) in cellTitles:
                ax.set_title(cellTitles[(r, c)])
            elif r == 0 and colHeaders is not None:
                ax.set_title(colHeaders[c])
            if c == 0 and rowHeaders is not None:
                # rotated and placed outside the y-axis label (fixed offset in points, so it stays
                # a consistent distance from the axis regardless of subplot size) rather than as a
                # title, which would sit in the gap shared with the row above and read as
                # ambiguously attached to neither row.
                ax.annotate(rowHeaders[r], xy=(0, 0.5), xytext=(-52, 0), xycoords="axes fraction",
                            textcoords="offset points", rotation=90, ha="center", va="center",
                            fontsize=plt.rcParams["axes.labelsize"])
            if c == 0:
                ax.set_ylabel(yLabel)
            if r == nRows - 1:
                ax.set_xlabel(xLabel)

    handles, labels = _legendHandles(legendEntries)
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False,
               bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure)
    fig.tight_layout()
    _save(fig, outputPathBase)
    plt.close(fig)


def renderSinglePanel(series, yLabel, xLabel, legendEntries, outputPathBase, ylim=None,
                       backgroundSpan=None, figSize=(6, 4)):
    fig, ax = plt.subplots(figsize=figSize)
    _plotCellSeries(ax, series, backgroundSpan=backgroundSpan)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_ylabel(yLabel)
    ax.set_xlabel(xLabel)
    handles, labels = _legendHandles(legendEntries)
    ax.legend(handles, labels, loc="best", frameon=False, ncol=2 if len(labels) > 4 else 1)
    fig.tight_layout()
    _save(fig, outputPathBase)
    plt.close(fig)


def renderStack(panels, yLabel, xLabel, legendEntries, outputPathBase, ylim=None,
                 backgroundSpan=None, panelHeaders=None, figSizePerPanel=(6, 2.2)):
    """
    A single column of stacked panels (e.g. noise_elevation_all's 1%/4% noise rows).
    panels: list of series-lists, one per panel (top to bottom).
    """
    nRows = len(panels)
    fig, axes = plt.subplots(nRows, 1, figsize=(figSizePerPanel[0], figSizePerPanel[1] * nRows),
                              sharex=True, squeeze=False, gridspec_kw={"hspace": 0.3})
    for r in range(nRows):
        ax = axes[r][0]
        _plotCellSeries(ax, panels[r], backgroundSpan=backgroundSpan)
        if ylim is not None:
            ax.set_ylim(ylim)
        if panelHeaders is not None:
            ax.annotate(panelHeaders[r], xy=(0, 0.5), xytext=(-52, 0), xycoords="axes fraction",
                        textcoords="offset points", rotation=90, ha="center", va="center",
                        fontsize=plt.rcParams["axes.labelsize"])
        ax.set_ylabel(yLabel)
        if r == nRows - 1:
            ax.set_xlabel(xLabel)

    handles, labels = _legendHandles(legendEntries)
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=False,
               bbox_to_anchor=(0.5, 0.0), bbox_transform=fig.transFigure)
    fig.tight_layout()
    _save(fig, outputPathBase)
    plt.close(fig)
