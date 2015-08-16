#!/bin/bash

python main.py --train_paths=/var/www/brat/data/finished --eval_with_cv --seed=3357830226 --pipeline_type=tregex --cv_folds=20 > ../outputs/tregex_20f.txt
python main.py --train_paths=/var/www/brat/data/finished --eval_with_cv --seed=3357830226 --pipeline_type=regex --cv_folds=20 > ../outputs/regex_20f.txt
python main.py --train_paths=/var/www/brat/data/finished --eval_with_cv --seed=3357830226 --pipeline_type=baseline+tregex --cv_folds=20 > ../outputs/tregex+baseline_20f.txt
python main.py --train_paths=/var/www/brat/data/finished --eval_with_cv --seed=3357830226 --pipeline_type=baseline+regex --cv_folds=20 > ../outputs/regex+baseline_20f.txt
python main.py --train_paths=/var/www/brat/data/finished --eval_with_cv --seed=3357830226 --pipeline_type=baseline --cv_folds=20 > ../outputs/baseline_20f.txt

python main.py --train_paths=/var/www/brat/data/Jeremy/PTB --eval_with_cv --seed=3357830226 --pipeline_type=regex --cv_folds=20 > ../outputs/ptb_regex_20f.txt
python main.py --train_paths=/var/www/brat/data/Jeremy/PTB --eval_with_cv --seed=3357830226 --pipeline_type=tregex --cv_folds=20 > ../outputs/ptb_tregex_20f.txt

python main.py --train_paths=/var/www/brat/data/Jeremy/PTB --eval_with_cv --seed=3357830226 --reader_gold_parses --pipeline_type=regex --cv_folds=20 > ../outputs/ptb_regex_20f_gold.txt
python main.py --train_paths=/var/www/brat/data/Jeremy/PTB --eval_with_cv --seed=3357830226 --reader_gold_parses --pipeline_type=tregex --cv_folds=20 > ../outputs/ptb_tregex_20f_gold.txt
