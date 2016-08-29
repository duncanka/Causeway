#!/bin/bash

SEED=2961393773
OUT_DIR=../outputs/final
DATA_DIR=/var/www/brat/data/finished
PTB_DATA_DIR=/var/www/brat/data/Jeremy/PTB
SHARED_FLAGS="--eval_with_cv --seed=$SEED --cv_folds=20"

export TS_SLOTS=4

# Columns:
# Run_type data_dir extra_flags
read -r -d '' PER_RUN_VARS << EOM
all3          $DATA_DIR
mostfreq      $DATA_DIR     --filter_classifiers=mostfreq
global        $DATA_DIR     --filter_classifiers=global
perconn       $DATA_DIR     --filter_classifiers=perconn
no_perconn    $DATA_DIR     --filter_classifiers=global,mostfreq
no_mostfreq   $DATA_DIR     --filter_classifiers=global,perconn
no_global     $DATA_DIR     --filter_classifiers=mostfreq,perconn
ptb_all3      $PTB_DATA_DIR
ptb_all3_gold $PTB_DATA_DIR --reader_gold_parses
EOM

tsp -n -L baseline bash -c "python main.py --train_paths=$DATA_DIR $SHARED_FLAGS --pipeline_type=baseline > $OUT_DIR/baseline.txt 2> $OUT_DIR/baseline.log"

for PIPELINE in regex tregex; do
    printf '%s\n' "$PER_RUN_VARS" | while IFS="\n" read line; do
        read NAME DIR FLAGS <<<$line
        echo -e "Pipeline:" $PIPELINE "\tRun type:" $NAME
        tsp -n -L $NAME bash -c "python main.py --train_paths=$DIR $SHARED_FLAGS --pipeline_type=$PIPELINE --models_dir=../models/${PIPELINE}_${NAME} $FLAGS > $OUT_DIR/${PIPELINE}_${NAME}.txt 2>$OUT_DIR/${PIPELINE}_${NAME}.log"
    done
    tsp -n -L "${PIPELINE}+baseline" bash -c "python main.py --train_paths=$DATA_DIR $SHARED_FLAGS  --pipeline_type=${PIPELINE}+baseline --models_dir=../models/${PIPELINE}+baseline $FLAGS > $OUT_DIR/${PIPELINE}+baseline.txt 2>$OUT_DIR/${PIPELINE}+baseline.log"
done

tsp -l