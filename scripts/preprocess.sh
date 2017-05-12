#!/bin/bash

# TODO: Make it use $STANFORD_PARSER_DIR for PARSERDIR if no second arg

if [ $# -ge 2 ]; then
    PARSERDIR=$2
elif [ $# -eq 0 ]; then
    echo "Usage: preprocess.sh input-dir [parser-dir]"
    exit;
else
    PARSERDIR=../../../stanford-parser-2015-04-20
    FILESDIR=$1
fi

find $FILESDIR -type f -name "*.txt" -exec "${0%/*}/cleanup.sh" {} \;
FILES=`find $FILESDIR -type f -not -name "*.*"`
java -mx4600m -cp "$PARSERDIR/classes:$PARSERDIR/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser \
    -tokenizerOptions "ptb3Escaping=false,normalizeParentheses=true,normalizeSpace=true,americanize=false,untokenizable=firstKeep,normalizeAmpersandEntity=true,normalizeFractions=true,ptb3Ellipsis=true" \
    -outputFormat "words,wordsAndTags,typedDependencies" -outputFormatOptions "nonCollapsedDependencies,stem" \
    -writeOutputFiles -outputFilesExtension "parse" -maxLength 275 \
    -printWordsForUnparsed edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz $FILES;
rm $FILES;
