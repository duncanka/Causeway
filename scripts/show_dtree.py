import os
from sklearn import tree
import subprocess
import sys
from nlpypline.pipeline.models import ClassBalancingClassifierWrapper

def vis_tree(model, tree_classifier=None, filename='/tmp/tree.dot',
             open_imgs=True):
    tree.export_graphviz(
        tree_classifier, out_file=filename, max_depth=20,
        feature_names=model.feature_name_dictionary.ids_to_names)

    fname_base = os.path.splitext(filename)[0]
    img_fname = "%s.png" % fname_base
    dot_proc = subprocess.Popen(["dot", "-Tpng", "-o%s" % img_fname, filename])
    (_std, err) = dot_proc.communicate()

    if err:
        sys.stderr.write(err)
    else:
        if open_imgs:
            with open(os.devnull, 'w') as null:
                subprocess.call(['xdg-open', img_fname], stderr=null)


def vis_pipeline_trees(pipeline, stage_num, open_imgs=True):
    model = pipeline.stages[stage_num].model
    if isinstance(model.classifier, ClassBalancingClassifierWrapper):
        classifier = model.classifier.classifier
    else:
        classifier = model.classifier

    if isinstance(classifier, tree.DecisionTreeClassifier):
        vis_tree(model, classifier)
    else:
        for i, estimator in enumerate(classifier.estimators_):
            vis_tree(model, estimator, '/tmp/tree_%d.dot' % i,
                     open_imgs)
