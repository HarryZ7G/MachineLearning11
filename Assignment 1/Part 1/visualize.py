import matplotlib.pyplot as plt

def visualize(x_s, y_s, labels, title, x_label, y_label, savefig=False, out_dir="."):
    """ This function generates multiple curves within a plot.

    Args:
    - x_s (list of lists): A list of x's for each curve.
    - y_s (list of lists): A list of y's for each curve.
    - labels (str): The labels for every curve.
    - title (str): The title of the plot.
    """
    assert len(x_s) == len(y_s) == len(labels)
    plt.clf()

    for (x, y, label) in zip(x_s, y_s, labels):
        plt.plot(x, y, label=label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    if savefig:
        fig = plt.gcf()
        fig.savefig(fname=f"{out_dir}/{title}.png", dpi=150, format="png")
    else:
        plt.show()


if __name__ == "__main__":
    import numpy as np
    x_s = [np.arange(10), np.arange(10)]
    y_s = [np.arange(10), np.arange(10) + 1]
    labels = ["1", "2"]
    title = "test"
    x_label = "x"
    y_label = "y"

    visualize(x_s, y_s, labels, title, x_label, y_label)
