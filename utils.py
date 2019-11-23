# encoding=utf8
import codecs
from collections import OrderedDict
import csv
import filecmp
import os
import re
import sys
import tempfile
import pdb

# reload(sys)
# sys.setdefaultencoding('utf8') 

BEGIN_BLOCK_COMMENT = '<!--\n'
END_BLOCK_COMMENT = '-->\n\n'
TRANSLATE_INDICATOR = '__translate the above block__\n'

def is_blank_line(line):
    return line.strip() == ''


def block_comment(input_md, output_md):
    last_line_is_blank = True
    with codecs.open(input_md, 'r', encoding='utf-8') as input_handle,\
            codecs.open(output_md, 'w', encoding='utf-8') as output_handle:
        for line in input_handle:
            this_line_is_blank = is_blank_line(line)
            if last_line_is_blank and not this_line_is_blank:
                output_handle.write(BEGIN_BLOCK_COMMENT)
            if this_line_is_blank:
                if last_line_is_blank:
                    # output_handle.write('\n')
                    continue
                output_handle.write(END_BLOCK_COMMENT)
                output_handle.write(TRANSLATE_INDICATOR)
                output_handle.write('\n')
                last_line_is_blank = True
            else:
                output_handle.write(line.replace(' -- ', ' \-\- '))
                last_line_is_blank = False

        output_handle.write(END_BLOCK_COMMENT)
        output_handle.write(TRANSLATE_INDICATOR)


if __name__ == '__main__':
    input_md = './chapter_preface/preface.md'
    output_md = './chapter_preface/preface_commented.md'
    block_comment(input_md, output_md)
