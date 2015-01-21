#!/usr/bin/env python

import re
import subprocess
import os
import sys
import httplib, urllib
from parse_to_dot import make_dot
from bs4 import BeautifulSoup

def extract_parse(html_doc):
    soup = BeautifulSoup(html_doc)
    return soup.body.find_all('pre')[1].getText()

def get_parser_response(sentence):
    cnxn = httplib.HTTPConnection("nlp.stanford.edu", 8080)
    params = urllib.urlencode({'query': sentence, 'parserSelect': 'English' })
    headers = {"Content-type": "application/x-www-form-urlencoded"}
    cnxn.request("POST", "/parser/index.jsp", params, headers)
    response = cnxn.getresponse()
    return response.read()



if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'sentence'
    tmp_file_path = '/tmp/%s.parse' % filename

    # Read from stdin
    print "Enter your sentence to parse below:"
    doc = get_parser_response(sys.stdin.readline())

    parse = extract_parse(doc)
    print parse
    print


    with open(tmp_file_path, 'w') as f:
        f.write(parse)

    with open(tmp_file_path, 'r') as f:
        make_dot(f)

    os.remove(tmp_file_path)
