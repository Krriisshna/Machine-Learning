import time
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
import matplotlib.colors as colors

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

def tlogistic_cost(X,y):
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

def soup_bowl():
    """ creates 3D quadratic error surface """
    #Create figure and plot with a 3D projection
    fig = plt.figure(figsize=(4,4))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    #Plot configuration
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_rotate_label(False)
    ax.view_init(15, -120)

    #Useful linearspaces to give values to the parameters w and b
    w = np.linspace(-20, 20, 100)
    b = np.linspace(-20, 20, 100)

    #Get the z value for a bowl-shaped cost function
    z=np.zeros((len(w), len(b)))
    j=0
    for x in w:
        i=0
        for y in b:
            z[i,j] = x**2 + y**2
            i+=1
        j+=1

    #Meshgrid used for plotting 3D functions
    W, B = np.meshgrid(w, b)

    #Create the 3D surface plot of the bowl-shaped cost function
    ax.plot_surface(W, B, z, cmap = "Spectral_r", alpha=0.7, antialiased=False)
    ax.plot_wireframe(W, B, z, color='k', alpha=0.1)
    ax.set_xlabel("$w$")
    ax.set_ylabel("$b$")
    ax.set_zlabel("Cost", rotation=90)
    ax.set_title("Squared Error Cost used in Linear Regression")

    plt.show()

def plt_logistic_squared_error(X,y):
    """ plots logistic squared error for demonstration """
    wx, by = np.meshgrid(np.linspace(-6,12,50),
                         np.linspace(10, -20, 40))
    points = np.c_[wx.ravel(), by.ravel()]
    cost = np.zeros(points.shape[0])

    for i in range(points.shape[0]):
        w,b = points[i]
        cost[i] = compute_cost_logistic_sq_err(X.reshape(-1,1), y, w, b)
    cost = cost.reshape(wx.shape)

    fig = plt.figure()
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(wx, by, cost, alpha=0.6,cmap=cm.jet,)

    ax.set_xlabel('w', fontsize=16)
    ax.set_ylabel('b', fontsize=16)
    ax.set_zlabel("Cost", rotation=90, fontsize=16)
    ax.set_title('"Logistic" Squared Error Cost vs (w, b)')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

def compute_cost_logistic_sq_err(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)                 #add sigmoid to normal sq error cost for linear regression
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2 * m)
    return np.squeeze(cost)

def compute_cost_matrix(X, y, w, b, logistic=False, lambda_=0, safe=True):
    m = X.shape[0]
    y = y.reshape(-1,1)             # ensure 2D
    w = w.reshape(-1,1)             # ensure 2D
    if logistic:
        if safe:  #safe from overflow
            z = X @ w + b                                                           #(m,n)(n,1)=(m,1)
            cost = -(y * z) + log_1pexp(z)
            cost = np.sum(cost)/m                                                   # (scalar)
        else:
            f    = sigmoid(X @ w + b)                                               # (m,n)(n,1) = (m,1)
            cost = (1/m)*(np.dot(-y.T, np.log(f)) - np.dot((1-y).T, np.log(1-f)))   # (1,m)(m,1) = (1,1)
            cost = cost[0,0]                                                        # scalar
    else:
        f    = X @ w + b                                                        # (m,n)(n,1) = (m,1)
        cost = (1/(2*m)) * np.sum((f - y)**2)                                   # scalar

    reg_cost = (lambda_/(2*m)) * np.sum(w**2)                                   # scalar

    total_cost = cost + reg_cost                                                # scalar

    return total_cost

def log_1pexp(x, maximum=20):
    out  = np.zeros_like(x,dtype=float)
    i    = x <= maximum
    ni   = np.logical_not(i)

    out[i]  = np.log(1 + np.exp(x[i]))
    out[ni] = x[ni]
    return out

def plt_tumor_data(x,y,ax):
    pos = y == 1
    neg = y == 0

    ax.scatter(x[pos],y[pos], s=80,marker='x',label='malignant',c='red')
    ax.scatter(x[neg],y[neg], s=100,marker='o',label='benign',facecolors='none',edgecolors=dlblue,lw=3)
    ax.set_ylim(-0.175,1.1)
    ax.set_ylabel('y')
    ax.set_xlabel('Tumor Size')
    ax.set_title("Logistic Regression on Categorical Data")

def gradient_descent(X, y, w_in, b_in, alpha, num_iters,
                      logistic=False, lambda_=0, verbose=True):
    """Batch gradient descent used by the interactive logistic plot."""
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    w = np.asarray(w_in, dtype=float).reshape(-1, 1).copy()
    b = float(b_in)
    m = X.shape[0]
    J_history = []

    for i in range(num_iters):
        z = X @ w + b

        if logistic:
            f = sigmoid(z)
        else:
            f = z

        dj_dw = (X.T @ (f - y)) / m
        dj_db = np.sum(f - y) / m

        if lambda_ != 0:
            dj_dw += (lambda_ / m) * w

        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i < 100000:
            J_history.append(
                compute_cost_matrix(
                    X, y, w, b,
                    logistic=logistic,
                    lambda_=lambda_,
                    safe=True
                )
            )

        if verbose and (i % max(1, num_iters // 10) == 0):
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.4f}")

    return w, b, J_history


class plt_quad_logistic:
    ''' plots a quad plot showing logistic regression '''
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-locals
    # pylint: disable=missing-function-docstring
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, x_train, y_train, w_range, b_range):
        fig = plt.figure(figsize=(10, 6))
        fig.canvas.toolbar_visible = False
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.set_facecolor('#ffffff')

        gs = GridSpec(2, 2, figure=fig)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0], projection='3d')
        ax3 = fig.add_subplot(gs[1, 1])

        pos = ax3.get_position().get_points()
        h = 0.05
        width = 0.2
        axcalc = plt.axes([pos[1, 0] - width, pos[1, 1] - h, width, h])
        ax = np.array([ax0, ax1, ax2, ax3, axcalc])

        self.fig = fig
        self.ax = ax
        self.x_train = x_train
        self.y_train = y_train

        self.w = 0.
        self.b = 0.

        self.dplot = data_plot(ax[0], x_train, y_train, self.w, self.b)
        self.con_plot = contour_and_surface_plot(
            ax[1], ax[2], x_train, y_train, w_range, b_range, self.w, self.b
        )
        self.cplot = cost_plot(ax[3])

        self.cid = fig.canvas.mpl_connect('button_press_event', self.click_contour)
        self.bcalc = Button(
            axcalc,
            'Run Gradient Descent \\nfrom current w,b (click)',
            color=dlc["dlorange"]
        )
        self.bcalc.on_clicked(self.calc_logistic)

    def click_contour(self, event):
        if event.inaxes == self.ax[1]:
            self.w = event.xdata
            self.b = event.ydata

            self.cplot.re_init()
            self.dplot.update(self.w, self.b)
            self.con_plot.update_contour_wb_lines(self.w, self.b)
            self.con_plot.path.re_init(self.w, self.b)

            self.fig.canvas.draw()

    def calc_logistic(self, event):
        for it in [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            w, self.b, J_hist = gradient_descent(
                self.x_train.reshape(-1, 1),
                self.y_train.reshape(-1, 1),
                np.array(self.w).reshape(-1, 1),
                self.b,
                0.1,
                it,
                logistic=True,
                lambda_=0,
                verbose=False
            )

            self.w = w[0, 0]
            self.dplot.update(self.w, self.b)
            self.con_plot.update_contour_wb_lines(self.w, self.b)
            self.con_plot.path.add_path_item(self.w, self.b)
            self.cplot.add_cost(J_hist)

            time.sleep(0.3)
            self.fig.canvas.draw()


class data_plot:
    ''' handles data plot '''
    # pylint: disable=missing-function-docstring
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, ax, x_train, y_train, w, b):
        self.ax = ax
        self.x_train = x_train
        self.y_train = y_train
        self.m = x_train.shape[0]
        self.w = w
        self.b = b

        self.plt_tumor_data()
        self.draw_logistic_lines(firsttime=True)
        self.mk_cost_lines(firsttime=True)

        self.ax.autoscale(enable=False)

    def plt_tumor_data(self):
        x = self.x_train
        y = self.y_train
        pos = y == 1
        neg = y == 0

        self.ax.scatter(
            x[pos], y[pos], marker='x', s=80, c='red', label="malignant"
        )
        self.ax.scatter(
            x[neg], y[neg], marker='o', s=100, label="benign",
            facecolors='none', edgecolors=dlc["dlblue"], lw=3
        )
        self.ax.set_ylim(-0.175, 1.1)
        self.ax.set_ylabel('y')
        self.ax.set_xlabel('Tumor Size')
        self.ax.set_title("Logistic Regression on Categorical Data")

    def update(self, w, b):
        self.w = w
        self.b = b
        self.draw_logistic_lines()
        self.mk_cost_lines()

    def draw_logistic_lines(self, firsttime=False):
        if not firsttime:
            self.aline[0].remove()
            self.bline[0].remove()
            self.alegend.remove()

        xlim = self.ax.get_xlim()
        x_hat = np.linspace(*xlim, 30)

        y_hat = sigmoid(np.dot(x_hat.reshape(-1, 1), self.w) + self.b)
        self.aline = self.ax.plot(
            x_hat, y_hat, color=dlc["dlblue"], label="y = sigmoid(z)"
        )

        f_wb = np.dot(x_hat.reshape(-1, 1), self.w) + self.b
        self.bline = self.ax.plot(
            x_hat, f_wb, color=dlc["dlorange"], lw=1,
            label=f"z = {np.squeeze(self.w):0.2f}x+({self.b:0.2f})"
        )
        self.alegend = self.ax.legend(loc='upper left')

    def mk_cost_lines(self, firsttime=False):
        if not firsttime:
            for artist in self.cost_items:
                artist.remove()

        self.cost_items = []
        cstr = f"cost = (1/{self.m})*("
        ctot = 0
        label = 'cost for point'
        addedbreak = False

        for p in zip(self.x_train, self.y_train):
            f_wb_p = sigmoid(self.w * p[0] + self.b)
            c_p = compute_cost_matrix(
                p[0].reshape(-1, 1),
                p[1],
                np.array(self.w),
                self.b,
                logistic=True,
                lambda_=0,
                safe=True
            )

            a = self.ax.vlines(
                p[0], p[1], f_wb_p, lw=3,
                color=dlc["dlpurple"], ls='dotted', label=label
            )
            label = ''

            cxy = [p[0], p[1] + (f_wb_p - p[1]) / 2]
            b = self.ax.annotate(
                f'{c_p:0.1f}',
                xy=cxy,
                xycoords='data',
                color=dlc["dlpurple"],
                xytext=(5, 0),
                textcoords='offset points'
            )

            cstr += f"{c_p:0.1f} +"
            if len(cstr) > 38 and not addedbreak:
                cstr += "\\n"
                addedbreak = True

            ctot += c_p
            self.cost_items.extend((a, b))

        ctot = ctot / len(self.x_train)
        cstr = cstr[:-1] + f") = {ctot:0.2f}"

        c = self.ax.text(
            0.05, 0.02, cstr,
            transform=self.ax.transAxes,
            color=dlc["dlpurple"]
        )
        self.cost_items.append(c)


class contour_and_surface_plot:
    ''' plots combined in class as they have similar operations '''
    # pylint: disable=missing-function-docstring
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, axc, axs, x_train, y_train, w_range, b_range, w, b):
        self.x_train = x_train
        self.y_train = y_train
        self.axc = axc
        self.axs = axs

        b_space = np.linspace(*b_range, 100)
        w_space = np.linspace(*w_range, 100)

        tmp_b, tmp_w = np.meshgrid(b_space, w_space)
        z = np.zeros_like(tmp_b)

        for i in range(tmp_w.shape[0]):
            for j in range(tmp_w.shape[1]):
                z[i, j] = compute_cost_matrix(
                    x_train.reshape(-1, 1),
                    y_train,
                    tmp_w[i, j],
                    tmp_b[i, j],
                    logistic=True,
                    lambda_=0,
                    safe=True
                )
                if z[i, j] == 0:
                    z[i, j] = 1e-9

        axc.contour(
            tmp_w, tmp_b, np.log(z),
            levels=12, linewidths=2, alpha=0.7, colors=dlcolors
        )
        axc.set_title('log(Cost(w,b))')
        axc.set_xlabel('w', fontsize=10)
        axc.set_ylabel('b', fontsize=10)
        axc.set_xlim(w_range)
        axc.set_ylim(b_range)

        self.update_contour_wb_lines(w, b, firsttime=True)

        axc.text(
            0.7, 0.05, "Click to choose w,b",
            bbox=dict(facecolor='white', ec='black'),
            fontsize=10,
            transform=axc.transAxes,
            verticalalignment='center',
            horizontalalignment='center'
        )

        axs.plot_surface(
            tmp_w, tmp_b, z, cmap=cm.jet,
            alpha=0.3, antialiased=True
        )
        axs.plot_wireframe(tmp_w, tmp_b, z, color='k', alpha=0.1)
        axs.set_xlabel("$w$")
        axs.set_ylabel("$b$")
        axs.zaxis.set_rotate_label(False)
        axs.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axs.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axs.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axs.set_zlabel("J(w, b)", rotation=90)
        axs.view_init(30, -120)

        axs.autoscale(enable=False)
        axc.autoscale(enable=False)

        self.path = path(self.w, self.b, self.axc)

    def update_contour_wb_lines(self, w, b, firsttime=False):
        self.w = w
        self.b = b

        cst = compute_cost_matrix(
            self.x_train.reshape(-1, 1),
            self.y_train,
            np.array(self.w),
            self.b,
            logistic=True,
            lambda_=0,
            safe=True
        )

        if not firsttime:
            for artist in self.dyn_items:
                artist.remove()

        a = self.axc.scatter(
            self.w, self.b, s=100, color=dlc["dlblue"],
            zorder=10, label="cost with \ncurrent w,b"
        )
        b_line = self.axc.hlines(
            self.b, self.axc.get_xlim()[0], self.w,
            lw=4, color=dlc["dlpurple"], ls='dotted'
        )
        c = self.axc.vlines(
            self.w, self.axc.get_ylim()[0], self.b,
            lw=4, color=dlc["dlpurple"], ls='dotted'
        )
        d = self.axc.annotate(
            f"Cost: {cst:0.2f}",
            xy=(self.w, self.b),
            xytext=(4, 4),
            textcoords='offset points',
            bbox=dict(facecolor='white'),
            size=10
        )
        e = self.axs.scatter3D(
            self.w, self.b, cst, marker='X', s=100
        )

        self.dyn_items = [a, b_line, c, d, e]


class path:
    ''' tracks paths during gradient descent on contour plot '''
    # pylint: disable=missing-function-docstring
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, w, b, ax):
        self.path_items = []
        self.w = w
        self.b = b
        self.ax = ax

    def re_init(self, w, b):
        for artist in self.path_items:
            artist.remove()
        self.path_items = []
        self.w = w
        self.b = b

    def add_path_item(self, w, b):
        a = FancyArrowPatch(
            posA=(self.w, self.b),
            posB=(w, b),
            color=dlc["dlblue"],
            arrowstyle='simple, head_width=5, head_length=10, tail_width=0.0',
        )
        self.ax.add_artist(a)
        self.path_items.append(a)
        self.w = w
        self.b = b


class cost_plot:
    """ manages cost plot for plt_quad_logistic """
    # pylint: disable=missing-function-docstring
    # pylint: disable=attribute-defined-outside-init
    def __init__(self,ax):
        self.ax = ax
        self.ax.set_ylabel("log(cost)")
        self.ax.set_xlabel("iteration")
        self.costs = []
        self.cline = self.ax.plot(0,0, color=dlc["dlblue"])

    def re_init(self):
        self.ax.clear()
        self.__init__(self.ax)

    def add_cost(self,J_hist):
        self.costs.extend(J_hist)
        self.cline[0].remove()
        self.cline = self.ax.plot(self.costs)

def plt_prob(ax, w_out,b_out):
    """ plots a decision boundary but include shading to indicate the probability """
    #setup useful ranges and common linspaces
    x0_space  = np.linspace(0, 4 , 100)
    x1_space  = np.linspace(0, 4 , 100)

    # get probability for x0,x1 ranges
    tmp_x0,tmp_x1 = np.meshgrid(x0_space,x1_space)
    z = np.zeros_like(tmp_x0)
    for i in range(tmp_x0.shape[0]):
        for j in range(tmp_x1.shape[1]):
            z[i,j] = sigmoid(np.dot(w_out, np.array([tmp_x0[i,j],tmp_x1[i,j]])) + b_out)


    cmap = plt.get_cmap('Blues')
    new_cmap = truncate_colormap(cmap, 0.0, 0.5)
    pcm = ax.pcolormesh(tmp_x0, tmp_x1, z,
                   norm=cm.colors.Normalize(vmin=0, vmax=1),
                   cmap=new_cmap, shading='nearest', alpha = 0.9)
    ax.figure.colorbar(pcm, ax=ax)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """ truncates color map """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def compute_cost_logistic(X, y, w, b, lambda_=0, safe=False):
    """
    Computes cost using logistic loss, non-matrix version

    Args:
      X (ndarray): Shape (m,n)  matrix of examples with n features
      y (ndarray): Shape (m,)   target values
      w (ndarray): Shape (n,)   parameters for prediction
      b (scalar):               parameter  for prediction
      lambda_ : (scalar, float) Controls amount of regularization, 0 = no regularization
      safe : (boolean)          True-selects under/overflow safe algorithm
    Returns:
      cost (scalar): cost
    """

    m,n = X.shape
    cost = 0.0
    for i in range(m):
        z_i    = np.dot(X[i],w) + b                                             #(n,)(n,) or (n,) ()
        if safe:  #avoids overflows
            cost += -(y[i] * z_i ) + log_1pexp(z_i)
        else:
            f_wb_i = sigmoid(z_i)                                                   #(n,)
            cost  += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)       # scalar
    cost = cost/m

    reg_cost = 0
    if lambda_ != 0:
        for j in range(n):
            reg_cost += (w[j]**2)                                               # scalar
        reg_cost = (lambda_/(2*m))*reg_cost

    return cost + reg_cost