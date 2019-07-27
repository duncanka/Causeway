#!/bin/bash
# Requires the Unix task spooler (package task-spooler in Ubuntu).
# Expects to be run from the Causeway root directory.

SEED=2961393773
OUT_DIR=outputs/final
LOG_DIR=$OUT_DIR/logs
BECAUSE_DIR=/home/jesse/Documents/BECAUSE
PTB_BECAUSE_DIR=$BECAUSE_DIR/PTB
STANFORD_DIR=/home/jesse/Documents/stanford-corenlp-full-2015-04-20/
# Max TRegex cache filename length is set to work with eCryptfs.
BASE_CMD="python2 src/causeway/main.py --eval_with_cv --seed=$SEED --cv_folds=20 
          --tregex_max_cache_filename_len=140 --tregex_dir=$STANFORD_DIR --stanford_ner_path=$STANFORD_DIR
          --reader_recurse --iaa_compute_overlapping=False" # --iaa_log_by_connective --iaa_log_by_category"

export PYTHONPATH="src:NLPypline/src"

# Columns:
# Run_type data_dir extra_flags
read -r -d '' PER_RUN_VARS << EOM
all3          $BECAUSE_DIR
mostfreq      $BECAUSE_DIR --filter_classifiers=mostfreq
global        $BECAUSE_DIR --filter_classifiers=global
perconn       $BECAUSE_DIR --filter_classifiers=perconn
no_perconn    $BECAUSE_DIR --filter_classifiers=global,mostfreq
no_mostfreq   $BECAUSE_DIR --filter_classifiers=global,perconn
no_global     $BECAUSE_DIR --filter_classifiers=mostfreq,perconn
no_world_knol $BECAUSE_DIR --filter_features_to_cancel=cause_ner:effect_ner,cause_hypernyms,effect_hypernyms,cause_lemma_skipgrams,effect_lemma_skipgrams
ptb_all3      $PTB_BECAUSE_DIR
ptb_all3_gold $PTB_BECAUSE_DIR --reader_gold_parses
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
run_pipeline tregex_cache tregex_cache $BECAUSE_DIR
run_pipeline tregex_cache tregex_cache_ptb $PTB_BECAUSE_DIR
NUM_CORES=$(grep -c ^processor /proc/cpuinfo)
tsp -n -L "parallelize" "tsp -S $NUM_CORES"

run_pipeline baseline baseline $BECAUSE_DIR
for PIPELINE_TYPE in tregex regex; do
    printf '%s\n' "$PER_RUN_VARS" | while IFS="\n" read line; do
        read RUN_TYPE DIR FLAGS <<<$line
        run_pipeline "$PIPELINE_TYPE" "${PIPELINE_TYPE}_${RUN_TYPE}" $DIR $FLAGS
    done
    run_pipeline "${PIPELINE_TYPE}+baseline" "${PIPELINE_TYPE}+baseline" $BECAUSE_DIR
    run_pipeline "${PIPELINE_TYPE}_mostfreq" "${PIPELINE_TYPE}_mostfreq_sep" $BECAUSE_DIR
done

tsp -l # Print spooled tasks
