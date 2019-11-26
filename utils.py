# encoding=utf8
import codecs
import filecmp
import re
import sys

# reload(sys)
# sys.setdefaultencoding('utf8')

BEGIN_BLOCK_COMMENT = '<!--\n'
END_BLOCK_COMMENT = '-->\n\n'
TRANSLATE_INDICATOR = '*translate the above block*\n'
HEADER_INDICATOR = ' *translate the above header*\n'
# Our special mark in markdown, e.g. :label:`chapter_intro`
MARK_RE_MD = re.compile(':([-\/\\._\w\d]+):`([\*-\/\\\._\w\d]+)`')


def is_blank_line(line):
    return line.strip() == ''


class MyLine(object):
    def __init__(self, line_str, in_code_block):
        self.line_str = line_str.replace(' -- ', ' \-\- ')
        self.in_code_block = in_code_block
        self.end_comment_if_next_line_blank = None

    def process(self, file_writer, last_line):
        if self.in_code_block:
            file_writer.write(self.line_str)
        else:
            self._process(file_writer, last_line)
        return self

    def _process(self, file_writer, last_line):
        raise NotImplementedError


class NormalLine(MyLine):
    def __init__(self, line_str, in_code_block):
        super(NormalLine, self).__init__(line_str, in_code_block)
        self.end_comment_if_next_line_blank = True

    def _process(self, file_writer, last_line):
        if last_line is None or isinstance(last_line, BlankLine):
            file_writer.write(BEGIN_BLOCK_COMMENT)
        file_writer.write(self.line_str)


class BlankLine(MyLine):
    def __init__(self, line_str, in_code_block):
        super(BlankLine, self).__init__(line_str, in_code_block)
        self.end_comment_if_next_line_blank = False

    def _process(self, file_writer, last_line):
        if isinstance(last_line, Header):
            file_writer.write('\n')
        elif last_line.end_comment_if_next_line_blank:
            file_writer.write(END_BLOCK_COMMENT)
            file_writer.write(TRANSLATE_INDICATOR)
            file_writer.write('\n')


class Header(MyLine):
    def __init__(self, line_str, in_code_block):
        super(Header, self).__init__(line_str, in_code_block)
        self.end_comment_if_next_line_blank = True
        self.heading = 0
        cnt = 0
        for char in self.line_str:
            if char == '#':
                cnt += 1
            elif char == ' ':
                self.heading = cnt
                break
            else:
                assert False, self.line_str

    def _process(self, file_writer, last_line):
        assert last_line is None or isinstance(last_line, BlankLine),\
            last_line.line_str
        file_writer.write(BEGIN_BLOCK_COMMENT)
        file_writer.write(self.line_str)
        file_writer.write(END_BLOCK_COMMENT)
        file_writer.write('#'*self.heading + HEADER_INDICATOR)


class IsCodeMarker(MyLine):
    pass


class IsMath(MyLine):
    pass


class IsLabel(MyLine):
    pass


class Line(object):
    def __init__(self, line_str, in_code_block=False):
        # since -- will close block comment
        self.line_str = line_str.replace(' -- ', ' \-\- ')
        if is_blank_line(line_str):
            self.line_type = BlankLine(line_str, in_code_block)
        else:
            self.line_type = NormalLine(line_str, in_code_block)

        # m = MARK_RE_MD.match(line_str)
        # self.is_label = m is not None and m[1] == 'label'
        # self.heading = 0
        # self.is_code_marker = line_str.startswith('```')
        # self.is_math = line_str.startswith('$')
        # if self.line_str.startswith('#'):
        #     cnt = 0
        #     for c in self.line_str:
        #         if c == '#':
        #             cnt += 1
        #         elif c == ' ':
        #             self.heading = cnt
        #             break
        #         else:
        #             assert False, self.line_str

    def process(self, file_writer, last_line):
        return self.line_type.process(file_writer, last_line)
        """last_line is a Line instance"""
        # if self.is_math:
        #     file_writer.write(self.line_str)
        #     file_writer.write('\n')
        #     return self

        # if in_code_block or self.is_code_marker:
        #     file_writer.write(self.line_str)
        #     return self

        # if self.is_blank_line:
        #     if last_line.is_blank_line or last_line.is_label or last_line.is_code_marker or last_line.is_math:
        #         return Line('')
        #     if last_line.heading > 0:
        #         file_writer.write('\n')
        #         return self
        #     file_writer.write(END_BLOCK_COMMENT)
        #     file_writer.write(TRANSLATE_INDICATOR)
        #     file_writer.write('\n')
        #     return self

        # if self.is_label:
        #     file_writer.write(self.line_str)
        #     file_writer.write('\n')
        #     return self

        # if self.heading > 0:
        #     file_writer.write(BEGIN_BLOCK_COMMENT)
        #     file_writer.write(self.line_str)
        #     file_writer.write(END_BLOCK_COMMENT)
        #     file_writer.write('#'*self.heading + HEADER_INDICATOR)
        #     return self

        # elif self.heading == 0:
        #     if last_line.is_blank_line:
        #         file_writer.write(BEGIN_BLOCK_COMMENT)
        #     file_writer.write(self.line_str.replace(' -- ', ' \-\- '))
        #     return self


def block_comment(input_md, output_md):
    last_line = None
    in_code_block = False
    with codecs.open(input_md, 'r', encoding='utf-8') as input_handle,\
            codecs.open(output_md, 'w', encoding='utf-8') as output_handle:
        for line_str in input_handle:
            this_line = Line(line_str, in_code_block)
            last_line = this_line.process(output_handle, last_line)
            # if this_line.is_code_marker:
            #     in_code_block = not in_code_block

        # if last_line.is_blank_line or last_line.is_label or last_line.is_code_marker:
        #     return
        output_handle.write(END_BLOCK_COMMENT)
        output_handle.write(TRANSLATE_INDICATOR)


if __name__ == '__main__':
    input_md = './chapter_preface/preface.md'
    output_md = './chapter_preface/preface_commented.md'
    block_comment(input_md, output_md)
