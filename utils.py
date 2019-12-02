# encoding=utf8
import codecs
import filecmp
import re
import sys
import argparse

# reload(sys)
# sys.setdefaultencoding('utf8')

BEGIN_BLOCK_COMMENT = '<!--\n'
END_BLOCK_COMMENT = '-->\n\n'
TRANSLATE_INDICATOR = '*dịch đoạn phía trên*'
HEADER_INDICATOR = ' *dịch tiêu đề phía trên*\n'
IMAGE_CAPTION_INDICATOR = '*dịch chú thích ảnh phía trên*'
START_FILE = '<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->\n'
END_FILE = '<!-- ===================== Kết thúc dịch Phần 1 ==================== -->\n'
SUFIX_PATH = 'contributors_template_vn.md'

# Our special mark in markdown, e.g. :label:`chapter_intro`
MARK_RE_MD = re.compile(':([-\/\\._\w\d]+):`([\*-\/\\\._\w\d]+)`')

parser = argparse.ArgumentParser(description='Dịch Dive into Deep Learning')
parser.add_argument('--convert', type=str, help='path to md file')


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
        if isinstance(last_line, BlankLine):
            file_writer.write(BEGIN_BLOCK_COMMENT)
        file_writer.write(self.line_str)


class BlankLine(MyLine):
    def __init__(self, line_str, in_code_block):
        super(BlankLine, self).__init__(line_str, in_code_block)
        self.end_comment_if_next_line_blank = False

    def _process(self, file_writer, last_line):
        if last_line.end_comment_if_next_line_blank:
            file_writer.write(END_BLOCK_COMMENT)
            file_writer.write(TRANSLATE_INDICATOR)
            file_writer.write('\n')
        file_writer.write('\n')


class HeaderLine(MyLine):
    def __init__(self, line_str, in_code_block):
        super(HeaderLine, self).__init__(line_str, in_code_block)
        self.end_comment_if_next_line_blank = False
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
        assert isinstance(last_line, BlankLine),\
            last_line.line_str
        file_writer.write(BEGIN_BLOCK_COMMENT)
        file_writer.write(self.line_str)
        file_writer.write(END_BLOCK_COMMENT)
        file_writer.write('#'*self.heading + HEADER_INDICATOR)


class ImageLine(MyLine):
    def __init(self, line_str, in_code_block):
        assert not in_code_block
        super(ImageLine, self).__init__(line_str, in_code_block)

    def _process(self, file_writer, last_line):
        close_square_bracket_id = self.line_str.index(']')
        assert self.line_str[close_square_bracket_id+1] == '(', self.line_str
        # assert self.line_str.endswith(')'), self.line_str
        file_writer.write(BEGIN_BLOCK_COMMENT)
        file_writer.write(self.line_str)
        file_writer.write(END_BLOCK_COMMENT)
        file_writer.write(
            '![' + IMAGE_CAPTION_INDICATOR + ']' + self.line_str[close_square_bracket_id+1:]
        )


class CodeMarkerLine(MyLine):
    def __init__(self, line_str, in_code_block):
        super(CodeMarkerLine, self).__init__(line_str, in_code_block)
        self.end_comment_if_next_line_blank = False

    def _process(self, file_writer, last_line):
        """ the print is printed in the super class"""
        file_writer.write(self.line_str)



class MathLine(MyLine):
    def __init__(self, line_str, in_code_block):
        super(MathLine, self).__init__(line_str, in_code_block)
        self.end_comment_if_next_line_blank = False

    def _process(self, file_writer, last_line):
        file_writer.write(self.line_str)
        return self


class LabelLine(MyLine):
    def __init__(self, line_str, in_code_block):
        super(LabelLine, self).__init__(line_str, in_code_block)
        self.end_comment_if_next_line_blank = False

    def _process(self, file_writer, last_line):
        # assert isinstance(last_line, HeaderLine) or isinstance(last_line, ImageLine), 'last line: {}\nthis_line: {}'.format(
        #         last_line.line_str, self.line_str
        #     )
        file_writer.write(self.line_str)
        # file_writer.write('\n')
        return self


def block_comment(input_md, output_md, add_prefix_suffix=False):
    last_line = BlankLine('', False)
    in_code_block = False
    with codecs.open(input_md, 'r', encoding='utf-8') as input_handle,\
            codecs.open(output_md, 'w', encoding='utf-8') as output_handle,\
            codecs.open(SUFIX_PATH, 'r', encoding='utf-8') as surfix_handle:
        if add_prefix_suffix:
            output_handle.write(START_FILE)
            output_handle.write('\n')
        for line_str in input_handle:
            line_str = line_str.rstrip() + '\n'
            line_str = line_str.replace(' -- ', ' \-\- ')
            match = MARK_RE_MD.match(line_str)
            if is_blank_line(line_str):
                line_type = BlankLine
            elif line_str.startswith('#'):
                line_type = HeaderLine
            elif line_str.startswith('!['):
                line_type = ImageLine
            elif line_str.startswith('$'):
                line_type = MathLine
            elif line_str.startswith('```'):
                in_code_block = not in_code_block
                line_type = CodeMarkerLine
            elif match is not None and match[1] in ['label', 'eqlabel']:
                line_type = LabelLine
            else:
                line_type = NormalLine

            this_line = line_type(line_str, in_code_block)
            last_line = this_line.process(output_handle, last_line)

        assert in_code_block is False

        # TODO: simplify 5 lines below
        if isinstance(last_line, BlankLine) or isinstance(last_line, LabelLine)\
                or isinstance(last_line, CodeMarkerLine) or isinstance(last_line, ImageLine):
            print('skip')
        else:
            output_handle.write(END_BLOCK_COMMENT)
            output_handle.write(TRANSLATE_INDICATOR)
        if add_prefix_suffix:
            output_handle.write('\n')
            output_handle.write(END_FILE)
            output_handle.write('\n')
            for line in surfix_handle:
                output_handle.write(line)


if __name__ == '__main__':
    args = parser.parse_args()
    input_md = args.convert
    output_md = input_md[:-len('.md')] + '_vn.md'
    block_comment(input_md, output_md, add_prefix_suffix=True)
