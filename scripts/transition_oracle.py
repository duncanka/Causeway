#!/usr/bin/env python

from gflags import FLAGS
from os.path import splitext
import sys

from causeway.because_data import (CausalityStandoffReader,
                                   CausalityOracleTransitionWriter)
from nlpypline.data.io import DirectoryReader

def main(argv):
    FLAGS(argv) # To avoid complaints
    files_dir = argv[1]
    
    reader = DirectoryReader((CausalityStandoffReader.FILE_PATTERN,),
                             CausalityStandoffReader(), True)
    reader.open(files_dir)
    documents = reader.get_all()
    
    writer = CausalityOracleTransitionWriter()
    for doc in documents:
        writer.open(splitext(doc.filename)[0] + '.trans')
        writer.write_all_instances(doc)

if __name__ == '__main__':
    main(sys.argv)