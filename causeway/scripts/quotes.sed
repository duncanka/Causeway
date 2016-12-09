#!/bin/sed -i.bak -f quotes.sed *.txt
s/ "/ ``/g
s/^"/``/g
s/("/(``/g
s/\t"/\t``/g # For .ann files

s/"$/''/g
s/" /'' /g
s/")/'')/g
s/",/'',/g
