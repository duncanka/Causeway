#!/bin/bash

SEED=2961393773
OUT_DIR=../outputs/final
DATA_DIR=/var/www/brat/data/finished
PTB_DATA_DIR=/var/www/brat/data/Jeremy/PTB
SHARED_FLAGS="--eval_with_cv --seed=$SEED --cv_folds=20"

python main.py --train_paths=$DATA_DIR $SHARED_FLAGS --pipeline_type=baseline                                                 > $OUT_DIR/baseline_20f.txt 2> $OUT_DIR/baseline_20f.log &

for PIPELINE_TYPE in regex tregex; do
    python main.py --train_paths=$DATA_DIR $SHARED_FLAGS --pipeline_type=$PIPELINE_TYPE                                       > $OUT_DIR/${PIPELINE_TYPE}_20f.txt 2>$OUT_DIR/${PIPELINE_TYPE}_20f.log &
    python main.py --train_paths=$DATA_DIR $SHARED_FLAGS --pipeline_type=$PIPELINE_TYPE --filter_classifiers=mostfreq         > $OUT_DIR/${PIPELINE_TYPE}_mostfreq_20f.txt 2> $OUT_DIR/${PIPELINE_TYPE}_mostfreq_20f.log &
    python main.py --train_paths=$DATA_DIR $SHARED_FLAGS --pipeline_type=$PIPELINE_TYPE --filter_classifiers=global           > $OUT_DIR/${PIPELINE_TYPE}_global_20f.txt 2> $OUT_DIR/${PIPELINE_TYPE}_global_20f.log &
    wait
    python main.py --train_paths=$DATA_DIR $SHARED_FLAGS --pipeline_type=$PIPELINE_TYPE --filter_classifiers=perconn          > $OUT_DIR/${PIPELINE_TYPE}_perconn_20f.txt 2> $OUT_DIR/${PIPELINE_TYPE}_perconn_20f.log &
    python main.py --train_paths=$DATA_DIR $SHARED_FLAGS --pipeline_type=$PIPELINE_TYPE --filter_classifiers=global,mostfreq  > $OUT_DIR/${PIPELINE_TYPE}_no_perconn_20f.txt 2> $OUT_DIR/${PIPELINE_TYPE}_no_perconn_20f.log &
    python main.py --train_paths=$DATA_DIR $SHARED_FLAGS --pipeline_type=$PIPELINE_TYPE --filter_classifiers=global,perconn   > $OUT_DIR/${PIPELINE_TYPE}_no_mostfreq_20f.txt 2> $OUT_DIR/${PIPELINE_TYPE}_no_mostfreq_20f.log &
    wait
    python main.py --train_paths=$DATA_DIR $SHARED_FLAGS --pipeline_type=$PIPELINE_TYPE --filter_classifiers=mostfreq,perconn > $OUT_DIR/${PIPELINE_TYPE}_no_global_20f.txt 2> $OUT_DIR/${PIPELINE_TYPE}_no_global_20f.log &
    python main.py --train_paths=$DATA_DIR $SHARED_FLAGS --pipeline_type=${PIPELINE_TYPE}+baseline                            > $OUT_DIR/${PIPELINE_TYPE}+baseline_20f.txt 2> $OUT_DIR/${PIPELINE_TYPE}+baseline_20f.log &
    python main.py --train_paths=$PTB_DATA_DIR $SHARED_FLAGS --pipeline_type=$PIPELINE_TYPE                                   > $OUT_DIR/ptb_${PIPELINE_TYPE}_20f.txt 2> $OUT_DIR/ptb_${PIPELINE_TYPE}_20f.log &
    python main.py --train_paths=$PTB_DATA_DIR $SHARED_FLAGS --pipeline_type=$PIPELINE_TYPE --reader_gold_parses              > $OUT_DIR/ptb_${PIPELINE_TYPE}_20f_gold.txt 2> $OUT_DIR/ptb_${PIPELINE_TYPE}_20f_gold.log &
    wait
done
