#!/bin/bash

# TODO: Make it use $STANFORD_PARSER_DIR for PARSERDIR if no second arg

FILESDIR=$1
if [ $# -ge 2 ]; then
    PARSERDIR=$2
else
    PARSERDIR=../../../stanford-parser-full-2014-10-31
fi

find $FILESDIR -type f -name "*.txt" -exec "${0%/*}/cleanup.sh" {} \;
FILES=`find $FILESDIR -type f -not -name "*.*"`
java -mx4600m -cp "$PARSERDIR/*:" edu.stanford.nlp.parser.lexparser.LexicalizedParser \
    -tokenizerOptions "ptb3Escaping=false,americanize=false,untokenizable=firstKeep" \
    -outputFormat "words,wordsAndTags,typedDependencies" -outputFormatOptions "nonCollapsedDependencies,stem" \
    -writeOutputFiles -outputFilesExtension "parse" -maxLength 275 \
    -printWordsForUnparsed edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz $FILES;
rm $FILES;
