import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np


def draw_vthresh(ax, x, left_color="#0096FF", right_color="#C00000"):
    if ax is None or not hasattr(ax, "get_xlim"):
        raise TypeError("ax must be a valid Matplotlib Axes object.")

    try:
        x = float(x)
    except (TypeError, ValueError) as error:
        raise ValueError("x must be a number.") from error

    x_left, x_right = ax.get_xlim()
    y_bottom, y_top = ax.get_ylim()

    ax.set_xlim(min(x_left, x), max(x_right, x))
    x_left, x_right = ax.get_xlim()

    y_middle = (y_bottom + y_top) / 2
    arrow_length = (x_right - x_left) * 0.08
    label_offset = (x_right - x_left) * 0.02

    ax.axvspan(x_left, x, color=left_color, alpha=0.20)
    ax.axvspan(x, x_right, color=right_color, alpha=0.20)
    ax.axvline(x=x, color="black", linestyle="--", linewidth=1.5)

    ax.annotate(
        "z >= 0",
        xy=(x + label_offset, y_middle),
        xytext=(8, 8),
        textcoords="offset points",
        color=right_color,
    )

    right_arrow = FancyArrowPatch(
        posA=(x, y_middle),
        posB=(x + arrow_length, y_middle),
        arrowstyle="simple,head_width=8,head_length=10,tail_width=1",
        color=right_color,
    )
    ax.add_patch(right_arrow)

    ax.annotate(
        "z < 0",
        xy=(x - label_offset, y_middle),
        xytext=(-35, 8),
        textcoords="offset points",
        ha="right",
        color=left_color,
    )

    left_arrow = FancyArrowPatch(
        posA=(x, y_middle),
        posB=(x - arrow_length, y_middle),
        arrowstyle="simple,head_width=8,head_length=10,tail_width=1",
        color=left_color,
    )
    ax.add_patch(left_arrow)

    return ax

def plot_data(X,y,ax, pos_label="y=1",neg_label="y=0", s=80,loc="best"):
    pos = y == 1
    neg = y == 0
    pos = pos.reshape(-1,)
    neg = neg.reshape(-1,)

    ax.scatter(X[pos,0],X[pos,1],marker='x',color='red',label=pos_label,s=s)
    ax.scatter(X[neg,0],X[neg,1],marker="o",label=neg_label,s=100,facecolors="none",lw=3,edgecolor="blue")
    ax.legend(loc=loc)

def sigmoid(z):
    z = np.clip(z, -500,500)
    # np.clip() limits the array to a chosen min. and max.
    # values below min. becomes min. and values above max. beacomes max.

    g = (1.0)/(1.0 + np.exp(-z))
    return g
z = np.array([1,2,3,4])
print(sigmoid(z))