import numpy as np
import matplotlib.pyplot as plt


def plot_saved_scores(causality_pipeline):
    model = causality_pipeline.stages[3].model
    saved_scores = model.decoder.saved
    positives, negatives = [
        [score for label, score in saved_scores if label == desired_label]
        for desired_label in [True, False]]

    plt.ylim([-2, 2])
    plt.xlim([-0.2, 1])
    classifier_names = ['Weighted', 'Per-connective', 'Most-frequent', 'Global']
    for i, classifier_name in zip(range(4), classifier_names):
        offset = .2 * (i + 1)
        pos = [p[i] for p in positives if not np.isnan(p[i])]
        plt.plot(pos, np.zeros_like(pos) + offset, 'bx',)
        plt.text(-0.2, offset, classifier_name)
        neg = [n[i] for n in negatives if not np.isnan(n[i])]
        plt.plot(neg, np.zeros_like(neg) - offset, 'r+',)
        plt.text(-.2, -offset, classifier_name)
    plt.show(block=False)
