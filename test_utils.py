# encoding=utf8
import codecs
import filecmp
import tempfile

from qcore.asserts import assert_eq

from utils import block_comment


def compare_files(filename1, filename2):
    with codecs.open(filename1, 'r', encoding='utf-8') as file1,\
            codecs.open(filename2, 'r', encoding='utf-8') as file2:
        cnt = 0
        for line1, line2 in zip(file1, file2):
            cnt += 1
            if line1 != line2:
                print(filename1)
                print('line {}: {} != {}'.format(cnt, line1, line2))
                return False
        if file1 or file2:
            print(file1 or file2)
            return False
    return True


class TestBlockComment(object):
    def check(self, input_md, output_md):
        temp_file_md = './tests/temp_output.md'
        block_comment(input_md, temp_file_md)
        print(filecmp.cmp(output_md, temp_file_md))
        assert filecmp.cmp(output_md, temp_file_md), compare_files(
            output_md, temp_file_md
        )

    def test_1(self):
        input_md = './tests/input_1.md'
        output_md = './tests/output_1.md'
        self.check(input_md, output_md)

    def test_2(self):
        """multiple blank lines"""
        input_md = './tests/input_2.md'
        output_md = './tests/output_2.md'
        self.check(input_md, output_md)

    def test_with_double_hythen(self):
        input_md = './tests/input_3.md'
        output_md = './tests/output_3.md'
        self.check(input_md, output_md)

    def test_with_label(self):
        input_md = './tests/input_4.md'
        output_md = './tests/output_4.md'
        self.check(input_md, output_md)

    def test_with_code_block(self):
        input_md = './tests/input_5.md'
        output_md = './tests/output_5.md'
        self.check(input_md, output_md)

    def test_header_no_label(self):
        input_md = './tests/input_6.md'
        output_md = './tests/output_6.md'
        self.check(input_md, output_md)

    def test_math(self):
        input_md = './tests/input_7.md'
        output_md = './tests/output_7.md'
        self.check(input_md, output_md)
