"""Confusion-matrix plotting utility used by tensorboard/wandb callbacks."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def _heatmap(data, row_labels, col_labels, axes=None, cbar_kw=None, cbarlabel="", **kwargs):
    if cbar_kw is None:
        cbar_kw = {}
    if not axes:
        axes = plt.gca()

    image = axes.imshow(data, **kwargs)
    cbar = axes.figure.colorbar(image, ax=axes, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    axes.set_xticks(np.arange(data.shape[1]))
    axes.set_yticks(np.arange(data.shape[0]))
    axes.set_xticklabels(col_labels)
    axes.set_yticklabels(row_labels)
    axes.set_ylabel("Actual Class")
    axes.set_xlabel("Predicted Class")
    axes.xaxis.set_label_position("top")
    axes.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(axes.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    for _, spine in axes.spines.items():
        spine.set_visible(False)

    axes.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    axes.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    axes.grid(which="minor", color="w", linestyle="-", linewidth=3)
    axes.tick_params(which="minor", bottom=False, left=False)

    return image, cbar


def _annotate_heatmap(image, data=None, valfmt="{x:.3f}", textcolors=("black", "white"), threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = image.get_array()

    if threshold is not None:
        threshold = image.norm(threshold)
    else:
        threshold = image.norm(data.max()) / 2.0

    kwargs = dict(horizontalalignment="center", verticalalignment="center")
    kwargs.update(textkw)

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    texts = []
    if isinstance(valfmt, (np.ndarray, np.generic)):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kwargs.update(color=textcolors[int(image.norm(data[i, j]) > threshold)])
                if valfmt[i, j] == "nodata":
                    text = image.axes.text(j, i, valfmt[i, j], **kwargs)
                else:
                    decimals = 1 if data[i, j] >= 1 else 3
                    text = image.axes.text(
                        j,
                        i,
                        str(np.round(data[i, j] / valfmt[i, j][0] if data[i, j] != 0 else 0, decimals))
                        + valfmt[i, j][1],
                        **kwargs,
                    )
                texts.append(text)
    else:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kwargs.update(color=textcolors[int(image.norm(data[i, j]) > threshold)])
                text = image.axes.text(j, i, valfmt(data[i, j], None), **kwargs)
                texts.append(text)
    return texts


def _cm_val_fmt(conf_mat, mark_no_data=False):
    def find_val_fmt(value):
        length_dict = {
            0: (10 ** 0, ""),
            3: (10 ** 3, "k"),
            6: (10 ** 6, "m"),
            9: (10 ** 9, "g"),
            12: (10 ** 12, "t"),
            15: (10 ** 15, "p"),
        }
        divider, unit_char = None, None
        for i, length in enumerate(length_dict):
            number = str(value).split(".")[0]
            if len(number) < length + 1:
                divider = length_dict[list(length_dict)[i - 1]][0]
                unit_char = length_dict[list(length_dict)[i - 1]][1]
                break
            elif len(number) == length + 1:
                divider = length_dict[length][0]
                unit_char = length_dict[length][1]
                break
            elif i == len(length_dict) - 1:
                divider = length_dict[list(length_dict)[i]][0]
                unit_char = length_dict[list(length_dict)[i]][1]
        return (divider, unit_char)

    cm_val_fmt = np.zeros_like(conf_mat, dtype=object)
    for i in range(conf_mat.shape[0]):
        if mark_no_data and all(np.equal(conf_mat[i], 0)):
            cm_val_fmt[i] = ["nodata" for _ in range(conf_mat.shape[1])]
        else:
            for j in range(conf_mat.shape[1]):
                cm_val_fmt[i, j] = find_val_fmt(conf_mat[i, j])
    return cm_val_fmt


def plot_confusion_matrix(conf_mat, labels, output_path=None, per_class_norm=False,
                          style=None, cmap="YlGn", figsize=None):
    """Plot a confusion matrix with the number of observation in the whole input dataset."""
    if style is not None:
        plt.style.use(style)

    if figsize is None:
        if conf_mat.shape[0] < 10:
            figsize = (10, 7)
        elif conf_mat.shape[0] <= 16:
            figsize = (12, 9)
        else:
            figsize = (16, 11)

    fig, axes = plt.subplots(figsize=figsize)
    cbarlabel = "Coefficients values"

    if per_class_norm:
        dividend = conf_mat.astype("float")
        divisor = conf_mat.sum(axis=1)[:, np.newaxis]
        conf_mat = np.divide(dividend, divisor, out=np.zeros_like(dividend), where=divisor != 0)

    image, _ = _heatmap(conf_mat, labels, labels, axes=axes, cmap=cmap, cbarlabel=cbarlabel)
    cm_val_fmt = _cm_val_fmt(conf_mat)
    _annotate_heatmap(image, valfmt=cm_val_fmt)

    fig.tight_layout(pad=3)

    if output_path is None:
        return fig
    plt.savefig(output_path)
    plt.close()
    return output_path
