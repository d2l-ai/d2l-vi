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
# Our special mark in markdown, e.g. :label:`chapter_intro`
MARK_RE_MD = re.compile(':([-\/\\._\w\d]+):`([\*-\/\\\._\w\d]+)`')


def is_blank_line(line):
    return line.strip() == ''


class Line(object):
    def __init__(self, line_str):
        self.is_header = False
        self.line_str = line_str
        self.is_blank_line = is_blank_line(line_str)
        m = MARK_RE_MD.match(line_str)
        self.is_label = m is not None and m[1] == 'label' 

    def process(self, file_writer, last_line):
        """last_line is a Line instance"""
        if last_line.is_blank_line and not self.is_blank_line:
            file_writer.write(BEGIN_BLOCK_COMMENT)
        if self.is_blank_line:
            if last_line.is_blank_line:
                return
            file_writer.write(END_BLOCK_COMMENT)
            file_writer.write(TRANSLATE_INDICATOR)
            file_writer.write('\n')
        else:
            file_writer.write(self.line_str.replace(' -- ', ' \-\- '))


def block_comment(input_md, output_md):
    last_line = Line('')
    with codecs.open(input_md, 'r', encoding='utf-8') as input_handle,\
            codecs.open(output_md, 'w', encoding='utf-8') as output_handle:
        for line_str in input_handle:
            this_line = Line(line_str)
            this_line.process(output_handle, last_line)
            last_line = this_line

        output_handle.write(END_BLOCK_COMMENT)
        output_handle.write(TRANSLATE_INDICATOR)


if __name__ == '__main__':
    input_md = './chapter_preface/preface.md'
    output_md = './chapter_preface/preface_commented.md'
    block_comment(input_md, output_md)
