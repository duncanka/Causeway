#!/bin/bash

# TODO: Make it use $STANFORD_PARSER_DIR for PARSERDIR if no second arg

if [ $# -ge 2 ]; then
    PARSERDIR=$2
elif [ $# -eq 0 ]; then
    echo "Usage: preprocess.sh input-dir [parser-dir]"
    exit;
else
    PARSERDIR=../../../stanford-parser
    FILESDIR=$1
fi

find $FILESDIR -type f -name "*.txt" -exec "${0%/*}/cleanup.sh" {} \;
FILES=`find $FILESDIR -type f -not -name "*.*"`
java -mx4600m -cp "$PARSERDIR/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser \
    -tokenizerOptions "ptb3Escaping=false,normalizeParentheses=true,americanize=false,untokenizable=firstKeep" \
    -outputFormat "words,wordsAndTags,typedDependencies,penn" -outputFormatOptions "nonCollapsedDependencies,stem" \
    -writeOutputFiles -outputFilesExtension "parse" -maxLength 275 \
    -printWordsForUnparsed edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz $FILES;
rm $FILES;
