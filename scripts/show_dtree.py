import os
import sklearn
from sklearn import tree
import subprocess

def vis_tree(model, tree_classifier=None, filename='/tmp/tree.dot'):
    sklearn.tree.export_graphviz(
        tree_classifier, out_file=filename, max_depth=20,
        feature_names=model.feature_name_dictionary.ids_to_names)

    fname_base = os.path.splitext(filename)[0]
    img_fname = "%s.png" % fname_base
    dot_proc = subprocess.Popen(["dot", "-Tpng", "-o%s" % img_fname, filename])
    (std, err) = dot_proc.communicate()
    if err:
        sys.stderr.write(err)
    else:
        with open(os.devnull, 'w') as null:
            subprocess.call(['xdg-open', img_fname], stderr=null)

def vis_pipeline_trees(pipeline):
    model = pipeline.stages[1].models[0]
    classifier = model.classifier.classifier
    if isinstance(classifier, tree.DecisionTreeClassifier):
        vis_tree(model, classifier)
    else:
        for i, estimator in classifier.estimators_:
            vis_tree(m, estimator, '/tmp/tree_%d.dot' % i)