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

run_pipeline() {
	PIPELINE=$1
	NAME=$2
	DIR=$3
	FLAGS=$4
    echo -e "Pipeline:" $PIPELINE "\tRun type:" $NAME
    tsp -n -L "$NAME" bash -c "$BASE_CMD --train_paths=$DIR --pipeline_type=$PIPELINE --models_dir='../models/$NAME' $FLAGS > '$OUT_DIR/$NAME.txt' 2> '$LOG_DIR/$NAME.log'"
}

mkdir -p $OUT_DIR
mkdir -p $LOG_DIR

run_pipeline baseline baseline $DATA_DIR

for PIPELINE_TYPE in tregex regex; do
    printf '%s\n' "$PER_RUN_VARS" | while IFS="\n" read line; do
        read RUN_TYPE DIR FLAGS <<<$line
        run_pipeline "$PIPELINE_TYPE" "${PIPELINE_TYPE}_${RUN_TYPE}" $DIR $FLAGS
    done
    run_pipeline "${PIPELINE_TYPE}+baseline" "${PIPELINE_TYPE}+baseline" $DIR
	run_pipeline "${PIPELINE_TYPE}_mostfreq" "${PIPELINE_TYPE}_mostfreq_sep" $DIR
done

tsp -l