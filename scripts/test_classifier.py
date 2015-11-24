import sys; sys.path.append('../scripts')

import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.io import loadmat
from sklearn import ensemble
from sklearn.pipeline import Pipeline as SKLPipeline
from sklearn.feature_selection.univariate_selection import SelectPercentile, \
    f_classif

from pipeline.models import ClassBalancingClassifierWrapper
from util.metrics import diff_binary_vectors, ClassificationMetrics
from sklearn.tree.tree import DecisionTreeClassifier
import vis_features


data = loadmat('../featurized_train.mat')
features_train = data['features_train']
labels_train = data['labels_train']
labels_train.shape = (labels_train.shape[1],)

data = loadmat('../featurized_test.mat')
features_test = data['features']
labels_test_gold = data['labels']
labels_test_gold.shape = (labels_test_gold.shape[1],)
# with open('../feature_names.pickle', 'r') as pickled:
#    feature_names = pickle.load(pickled)

print "Loaded data; testing classifier..."

features_train, labels_train = ClassBalancingClassifierWrapper.rebalance(
    features_train, labels_train, ratio=2)


results = []
for i in range(15):
    print 'Round', i
    classifier = DecisionTreeClassifier()
    classifier = SKLPipeline([
       ('feature_selection', SelectPercentile(f_classif, 1)),
       ('classification', classifier)
    ])
    classifier.fit(features_train, labels_train)

    labels_test_predicted = classifier.predict(features_test)
    results.append(diff_binary_vectors(labels_test_predicted, labels_test_gold))

# support = classifier.steps[0][1].get_support(True)
# print 'Selected', len(support), 'features:'
# for index in support:
#    print '   ', feature_names[index]

print 'Results:'
print ClassificationMetrics.average(results, False)


# Visualize last round
'''
fig = plt.figure()
fig.canvas.set_window_title('All training features')
vis_features.vis_features(features_train, labels_train)

selected = classifier.steps[0][1].transform(features_train)
fig = plt.figure()
fig.canvas.set_window_title('All selected training features')
vis_features.vis_features(selected, labels_train)
'''

selected = classifier.steps[0][1].transform(features_test)
for predicted_class, win_title in [(1, 'TPs/FPs'), (0, 'TNs/FNs')]:
    indices_in_class = np.where(labels_test_predicted == predicted_class)[0]
    selected_in_class = selected[indices_in_class]
    correct = (labels_test_gold[indices_in_class] ==
               labels_test_predicted[indices_in_class])
    fig = plt.figure()
    fig.canvas.set_window_title(win_title)
    # Correctly labeled samples are indicated by 1's, so will be on top.
    vis_features.vis_features(selected_in_class, correct)
