import os
import matplotlib
matplotlib.use('Agg') # set under "import matplotlib" line
import matplotlib.pyplot as plt


def plot_history(save_path, history):
    """plot the history of accuracy transition.

    Args:
        save_path (str): Path to save image directory.
        history (int): path to history file.

    Returns:
        A visualization image that plotting learning history
    """
    plt.plot(history['acc'], marker='.')
    plt.plot(history['val_acc'], '--.')
    plt.plot(history['loss'], marker='.')
    plt.plot(history['val_loss'], '--.')

    plt.grid()
    plt.xlabel('epoch')
    plt.legend(['acc', 'val_acc', 'loss', 'val_loss'], loc='best')
    filename = "learning_history.png"
    plot_img_path = os.path.join(save_path, filename)
    plt.savefig(plot_img_path)
