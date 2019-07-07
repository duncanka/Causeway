#!/bin/bash

if [ $# -ge 3 ]; then
    PARSERDIR=$3
elif [ $# -lt 2 ]; then
    echo "Usage: convert-mrg.sh txt-input-dir mrg-input-dir [parser-dir]"
    exit;
else
    PARSERDIR=../../../stanford-parser-2015-04-20/
fi

TXT_DIR=$1
MRG_DIR=$2

for f in $TXT_DIR/*.txt; do
    base_name=${f%.txt}
    out="$base_name.parse.gold"
    echo "$f to $out"
    base_name=${base_name##*/}
    java -mx800m -cp "$PARSERDIR/classes:$PARSERDIR/*:" edu.stanford.nlp.trees.TreePrint \
         -format 'words,wordsAndTags,typedDependencies,penn' -options 'nonCollapsedDependencies,stem' \
         -norm "edu.stanford.nlp.trees.BobChrisTreeNormalizer" $MRG_DIR/${base_name:0:2}/$base_name.mrg > $out
done
