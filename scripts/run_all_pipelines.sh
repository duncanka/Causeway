#!/bin/bash

SEED=2961393773
OUT_DIR=../outputs/final
LOG_DIR=$OUT_DIR/logs
DATA_DIR=/var/www/brat/data/finished
PTB_DATA_DIR=/var/www/brat/data/Jeremy/PTB
BASE_CMD="python main.py --eval_with_cv --seed=$SEED --cv_folds=20 --iaa_log_by_connective --iaa_log_by_category"

tsp -S 4

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

echo -e "Pipeline: baseline"
tsp -n -L baseline bash -c "$BASE_CMD --train_paths=$DATA_DIR --pipeline_type=baseline > $OUT_DIR/baseline.txt 2> $LOG_DIR/baseline.log"

for PIPELINE in regex tregex; do
    printf '%s\n' "$PER_RUN_VARS" | while IFS="\n" read line; do
        read NAME DIR FLAGS <<<$line
        echo -e "Pipeline:" $PIPELINE "\tRun type:" $NAME
        tsp -n -L "${PIPELINE} ${NAME}" bash -c "$BASE_CMD --train_paths=$DIR --pipeline_type=$PIPELINE --models_dir=../models/${PIPELINE}_${NAME} $FLAGS > $OUT_DIR/${PIPELINE}_${NAME}.txt 2> $LOG_DIR/${PIPELINE}_${NAME}.log"
    done
    tsp -n -L "${PIPELINE}+baseline" bash -c "$BASE_CMD --train_paths=$DATA_DIR --pipeline_type=${PIPELINE}+baseline --models_dir=../models/${PIPELINE}+baseline > $OUT_DIR/${PIPELINE}+baseline.txt 2> $LOG_DIR/${PIPELINE}+baseline.log"
    tsp -n -L "${PIPELINE} mostfreq_sep" bash -c "$BASE_CMD --train_paths=$DATA_DIR --pipeline_type=${PIPELINE}+baseline --models_dir=../models/${PIPELINE}_mostfreq > $OUT_DIR/${PIPELINE}_mostfreq_sep.txt 2> $LOG_DIR/${PIPELINE}_mostfreq_sep.log"
done

tsp -l