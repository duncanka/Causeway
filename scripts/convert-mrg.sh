#!/bin/bash

if [ $# -ge 2 ]; then
    PARSERDIR=$2
elif [ $# -eq 0 ]; then
    echo "Usage: convert-mrg.sh input-dir [parser-dir]"
    exit;
else
    PARSERDIR=../../../stanford-parser
fi

for f in $1/*.mrg; do
    base_name=${f%.mrg}
    out="$base_name.parse.gold"
    echo "$f to $out"
    java -mx800m -cp "$PARSERDIR/*:" edu.stanford.nlp.trees.TreePrint \
         -format 'words,wordsAndTags,typedDependencies,penn' -options 'nonCollapsedDependencies,stem' \
         -norm "edu.stanford.nlp.trees.BobChrisTreeNormalizer" $f > $out
done
