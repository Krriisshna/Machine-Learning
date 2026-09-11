import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

dlc = dict(dlblue = '#0096ff', dlorange = '#FF9300', dldarkred='#C00000', dlmagenta='#FF40FF', dlpurple='#7030A0')
dlblue = '#0096ff'; dlorange = '#FF9300'; dldarkred='#C00000'; dlmagenta='#FF40FF'; dlpurple='#7030A0'
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]


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

def plt_logistic_cost(X,y):
    """ plots logistic cost """
    wx, by = np.meshgrid(np.linspace(-6,12,50),
                         np.linspace(0, -20, 40))
    points = np.c_[wx.ravel(), by.ravel()]
    cost = np.zeros(points.shape[0],dtype=np.longdouble)

    for i in range(points.shape[0]):
        w,b = points[i]
        cost[i] = compute_cost_matrix(X.reshape(-1,1), y, w, b, logistic=True, safe=True)
    cost = cost.reshape(wx.shape)

    fig = plt.figure(figsize=(9,5))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(wx, by, cost, alpha=0.6,cmap=cm.jet,)

    ax.set_xlabel('w', fontsize=16)
    ax.set_ylabel('b', fontsize=16)
    ax.set_zlabel("Cost", rotation=90, fontsize=16)
    ax.set_title('Logistic Cost vs (w, b)')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_surface(wx, by, np.log(cost), alpha=0.6,cmap=cm.jet,)

    ax.set_xlabel('w', fontsize=16)
    ax.set_ylabel('b', fontsize=16)
    ax.set_zlabel('\nlog(Cost)', fontsize=16)
    ax.set_title('log(Logistic Cost) vs (w, b)')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    plt.show()
    return cost

def plt_two_logistic_loss_curves():
    """ plots the logistic loss """
    fig,ax = plt.subplots(1,2,figsize=(6,3),sharey=True)
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    x = np.linspace(0.01,1-0.01,20)
    ax[0].plot(x,-np.log(x))
    #ax[0].set_title("y = 1")
    ax[0].text(0.5, 4.0, "y = 1", fontsize=12)
    ax[0].set_ylabel("loss")
    ax[0].set_xlabel(r"$f_{w,b}(x)$")
    ax[1].plot(x,-np.log(1-x))
    #ax[1].set_title("y = 0")
    ax[1].text(0.5, 4.0, "y = 0", fontsize=12)
    ax[1].set_xlabel(r"$f_{w,b}(x)$")
    ax[0].annotate("prediction \nmatches \ntarget ", xy= [1,0], xycoords='data',
                 xytext=[-10,30],textcoords='offset points', ha="right", va="center",
                   arrowprops={'arrowstyle': '->', 'color': dlorange, 'lw': 3},)
    ax[0].annotate("loss increases as prediction\n differs from target", xy= [0.1,-np.log(0.1)], xycoords='data',
                 xytext=[10,30],textcoords='offset points', ha="left", va="center",
                   arrowprops={'arrowstyle': '->', 'color': dlorange, 'lw': 3},)
    ax[1].annotate("prediction \nmatches \ntarget ", xy= [0,0], xycoords='data',
                 xytext=[10,30],textcoords='offset points', ha="left", va="center",
                   arrowprops={'arrowstyle': '->', 'color': dlorange, 'lw': 3},)
    ax[1].annotate("loss increases as prediction\n differs from target", xy= [0.9,-np.log(1-0.9)], xycoords='data',
                 xytext=[-10,30],textcoords='offset points', ha="right", va="center",
                   arrowprops={'arrowstyle': '->', 'color': dlorange, 'lw': 3},)
    plt.suptitle("Loss Curves for Two Categorical Target Values", fontsize=12)
    plt.tight_layout()
    plt.show()

    def plt_simple_example(x,y):
        pos = y == 1
        neg = y == 0

        fig,ax = plt.subplots(1,1, figsize=(5,3))
        ax.scatter(x[pos],y[pos],marker='x',s=80,label='malignant',color='red')
        ax.scatter(x[neg],y[neg],marker='o',s=100,label='benign',facecolor='none',edgecolors=dlblue,lw=3)
        ax.set_ylim(-0.075,1.1)
        ax.set_ylabel('y')
        ax.set_xlabel('tumor size')
        ax.legend(loc='lower right')
        ax.set_title("Example of Logistic Regression on Categorical Data")