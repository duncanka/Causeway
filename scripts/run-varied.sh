#!/bin/bash

for PCT in 60 70 80 90; do
    python main.py --train_paths=/var/www/brat/data/finished-$PCT-percent/ --eval_with_cv \
    	--pipeline_type=$1 --cv_folds=20 --cv_print_fold_results=0 >> ../outputs/$1_20f_varied_pct.txt;
done
