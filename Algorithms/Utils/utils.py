# Utils for Algorithms

# Import relevant libraries
import os
import os

import numpy as np
import csv
import matplotlib


def save_plot(fig, path="plots", tight_layout=True, fig_extension="png", resolution=300):
    """
        Function for saving figures and plots
        :arg
            1. fig: label of the figure
            2. path (optional): output path of the figure
    """

    fig_path = os.path.join(".", path, fig + "." + fig_extension)

    print("Saving figure...", fig)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_path, format=fig_extension, dpi=resolution)
    print("figure can be found in: ", path)


# Script for saving the output
decision_scorePath = "./"
def write_decisionScores2Csv(path, filename, positiveScores, negativeScores):
    newfilePath = path + filename
    print("Writing file to ", path + filename)
    poslist = positiveScores.tolist()
    neglist = negativeScores.tolist()

    # rows = zip(poslist, neglist)
    d = [poslist, neglist]
    export_data = zip_longest(*d, fillvalue='')
    with open(newfilePath, 'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("Training", "Testing"))
        wr.writerows(export_data)
    myfile.close()

    return


