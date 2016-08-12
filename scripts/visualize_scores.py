import numpy as np
import matplotlib.pyplot as plt


def plot_saved_scores(causality_pipeline):
    model = causality_pipeline.stages[3].model
    saved_scores = model.decoder.saved
    positives, negatives = [
        [score for label, score in saved_scores if label == desired_label]
        for desired_label in [True, False]]

    plt.plot(positives, np.zeros_like(positives) + .2, 'bx',)
    plt.plot(negatives, np.zeros_like(negatives) - .2, 'r+',)
    plt.ylim([-2, 2])
    plt.show(block=False)
