#!/bin/bash


# TODO: Figure out how to make this eliminate the section headers with
# biographical/date information in hearing documents

#sed -n '
echo '
  # Only read up to the APPENDIX line
  1,/A P P E N D I X/ {
    # Ignore the APPENDIX line itself
    /A P P E N D I X/ !{
       # Remove footnotes
       s/\\[0-9][0-9]*\\/ /g
       # Remove weird strings of formatting dashes/underscores/dots that confuse the parser
       s/[-_\.]\{5,\}/ /g
       # Turn ellipses back to normal
       s/ \. \. \./ ... /g
       # Replace question labels with periods (avoids combining them with previous word, too)
       s/Q\.[0-9][0-9]*\(\.[a-z]\)\{0,1\}\././g
       # Delete answer labels
       s/A\.[0-9][0-9]*\./ /g
       # Sentencify endings of lines with interrupted "----" sequences
       s/\([A-Za-z0-9]-\{2,4\}\)$/\1./g
       # Sentencify endings of lines that are ALL CAPS (metadata about source of info; confuses parser)
       s/\([A-Z][A-Z,\. ]\{9,\}\)$/'"\n"'\1./
       # Sentencify lines that are just the name of the speaker (so as not to confuse the parser)
       s/^\([A-Z][A-Za-z\. ]\{1,55\}\):$/\1./
       # Correct titles that are stuck on the end of sentences
       # (pattern: some alphabetic text, followed by a period, followed immediately by
       # up to 9 words, followed by an optional question mark)
       s/\([A-Za-z]\{3,\}\)\.\(\([A-Z][A-Za-z]*\)\( [A-Z][A-Za-z]*\)\{1,8\}?\{0,1\}\)$/\1.'"\n\n"'\2./g
       p
    }
  }
' > /dev/null
cat $1 > ${1%.*} # Output to a temporary file that the parser will actually parse.
