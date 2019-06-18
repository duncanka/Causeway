#!/bin/bash
# Requires the Unix task spooler (package task-spooler in Ubuntu).
# Expects to be run from the Causeway root directory.

SEED=2961393773
OUT_DIR=outputs/final
LOG_DIR=$OUT_DIR/logs
ALL_DATA_DIR=/var/www/brat/data/finished
PTB_DATA_DIR=/var/www/brat/data/Jeremy/PTB
STANFORD_DIR=/home/jesse/Documents/Work/Research/stanford-parser-full-2015-04-20/
# Set max TRegex cache filename length to work with eCryptfs.
BASE_CMD="python2 src/causeway/main.py --eval_with_cv --seed=$SEED --cv_folds=20 --iaa_log_by_connective --iaa_log_by_category --tregex_max_cache_filename_len=140 --tregex_dir=$STANFORD_DIR --stanford_ner_path=$STANFORD_DIR"

export PYTHONPATH="src:NLPypline/src"

# Columns:
# Run_type data_dir extra_flags
read -r -d '' PER_RUN_VARS << EOM
all3          $ALL_DATA_DIR
mostfreq      $ALL_DATA_DIR --filter_classifiers=mostfreq
global        $ALL_DATA_DIR --filter_classifiers=global
perconn       $ALL_DATA_DIR --filter_classifiers=perconn
no_perconn    $ALL_DATA_DIR --filter_classifiers=global,mostfreq
no_mostfreq   $ALL_DATA_DIR --filter_classifiers=global,perconn
no_global     $ALL_DATA_DIR --filter_classifiers=mostfreq,perconn
no_world_knol $ALL_DATA_DIR --filter_features_to_cancel=cause_ner:effect_ner,cause_hypernyms,effect_hypernyms,cause_lemma_skipgrams,effect_lemma_skipgrams
ptb_all3      $PTB_DATA_DIR
ptb_all3_gold $PTB_DATA_DIR --reader_gold_parses
EOM

run_pipeline() {
	PIPELINE=$1
	NAME=$2
	DIR=$3
	FLAGS=$4
    echo -e "Pipeline:" $PIPELINE "\tRun type:" $NAME
    tsp -n -L "$NAME" bash -c "$BASE_CMD --train_paths=$DIR --pipeline_type=$PIPELINE --models_dir='models/$NAME' $FLAGS > '$OUT_DIR/$NAME.txt' 2> '$LOG_DIR/$NAME.log'"
}

mkdir -p $OUT_DIR
mkdir -p $LOG_DIR

tsp -S 1 # for TRegex caching
run_pipeline tregex_cache tregex_cache $ALL_DATA_DIR
run_pipeline tregex_cache tregex_cache_ptb $PTB_DATA_DIR
tsp -n -L "parallelize" "tsp -S 4"

run_pipeline baseline baseline $ALL_DATA_DIR
for PIPELINE_TYPE in tregex regex; do
    printf '%s\n' "$PER_RUN_VARS" | while IFS="\n" read line; do
        read RUN_TYPE DIR FLAGS <<<$line
        run_pipeline "$PIPELINE_TYPE" "${PIPELINE_TYPE}_${RUN_TYPE}" $DIR $FLAGS
    done
    run_pipeline "${PIPELINE_TYPE}+baseline" "${PIPELINE_TYPE}+baseline" $ALL_DATA_DIR
    run_pipeline "${PIPELINE_TYPE}_mostfreq" "${PIPELINE_TYPE}_mostfreq_sep" $ALL_DATA_DIR
done

tsp -l # Print spooled tasks
