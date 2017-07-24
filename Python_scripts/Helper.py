from matplotlib import pyplot as plt
import numpy as np


class Helper:
    @staticmethod
    def fancy_plot_confusion_matrix(cm, target_names, title='Confusion matrix',
                                    cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.tight_layout()

        width, height = cm.shape

        for x in range(width):
            for y in range(height):
                plt.annotate(str(cm[x][y]), xy=(y, x),
                             horizontalalignment='center',
                             verticalalignment='center')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
