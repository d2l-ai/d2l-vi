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
                print('line {}: {} != {}'.format(cnt, line1, line2))
                return False
        if file1 or file2:
            print(file1 or file2)
            return False
    return True


class TestBlockComment(object):
    def test_1(self):
        input_md = './tests/input_1.md'
        output_md = './tests/output_1.md'
        with tempfile.NamedTemporaryFile() as temp:
            block_comment(input_md, temp.name)
            print(filecmp.cmp(output_md, temp.name))
            assert filecmp.cmp(output_md, temp.name), compare_files(
                output_md, temp.name
            )

    def test_2(self):
        """multiple blank lines"""
        input_md = './tests/input_2.md'
        output_md = './tests/output_2.md'
        with tempfile.NamedTemporaryFile() as temp:
            block_comment(input_md, temp.name)
            print(filecmp.cmp(output_md, temp.name))
            assert filecmp.cmp(output_md, temp.name), compare_files(
                output_md, temp.name
            )

    def test_with_double_hythen(self):
        input_md = './tests/input_3.md'
        output_md = './tests/output_3.md'
        with tempfile.NamedTemporaryFile() as temp:
            block_comment(input_md, temp.name)
            print(filecmp.cmp(output_md, temp.name))
            assert filecmp.cmp(output_md, temp.name), compare_files(
                output_md, temp.name
            )

    def test_with_header(self):
        input_md = './tests/input_4.md'
        output_md = './tests/output_4.md'
        temp_ouput_md = './tests/temp_ouput.md'
        # with tempfile.NamedTemporaryFile() as temp:
        block_comment(input_md, temp_ouput_md)
        print(filecmp.cmp(output_md, temp_ouput_md))
        assert filecmp.cmp(output_md, temp_ouput_md), compare_files(
            output_md, temp_ouput_md
        )

    # def test_with_code_block(self):
    #     input_md = './tests/input_5.md'
    #     output_md = './tests/output_5.md'
    #     with tempfile.NamedTemporaryFile() as temp:
    #         block_comment(input_md, temp.name)
    #         print(filecmp.cmp(output_md, temp.name))
    #         assert filecmp.cmp(output_md, temp.name), compare_files(
    #             output_md, temp.name
    #         )
