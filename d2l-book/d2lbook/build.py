import os
import sys
import notedown
import nbformat
import nbconvert
import pkg_resources
import logging
import shutil
import time
import datetime
import argparse
import re
import regex
import subprocess
import hashlib
from d2lbook.utils import *
from d2lbook.sphinx import prepare_sphinx_env
from d2lbook.config import Config
from d2lbook import colab, sagemaker

__all__  = ['build']

# Our special mark in markdown, e.g. :label:`chapter_intro`
mark_re_md = re.compile(':([-\/\\._\w\d]+):`([\*-\/\\\._\w\d]+)`')
# Same as mark_re_md for rst, while ` is replaced with `` in mark_re
mark_re = re.compile(':([-\/\\._\w\d]+):``([\*-\/\\\._\w\d]+)``')

commands = ['eval', 'rst', 'html', 'pdf', 'pkg', 'linkcheck', 'ipynb',
            'outputcheck', 'lib', 'colab', 'sagemaker', 'all']

def build():
    parser = argparse.ArgumentParser(description='Build the documents')
    parser.add_argument('commands', nargs='+', choices=commands)
    args = parser.parse_args(sys.argv[2:])
    config = Config()
    builder = Builder(config)
    for cmd in args.commands:
        getattr(builder, cmd)()

class Builder(object):
    def __init__(self, config):
        self.config = config
        mkdir(self.config.tgt_dir)
        self.sphinx_opts = '-j 4'
        if config.build['warning_is_error'].lower() == 'true':
            self.sphinx_opts += ' -W'
        self.done = dict((cmd, False) for cmd in commands)

    def _once(func):
        # An decorator that run a method only once
        def warp(self):
            name = func.__name__
            if self.done[name]:
                return
            full_name = 'd2lbook build ' + name
            tik = datetime.datetime.now()
            func(self)
            logging.info('=== Finished "%s" in %s', full_name,
                         get_time_diff(tik, datetime.datetime.now()))
            self.done[name] = True
        return warp

    def _find_md_files(self):
        build = self.config.build
        src_dir = self.config.src_dir
        notebooks = find_files(build['notebooks'], src_dir,
                               build['exclusions']+' '+build['non-notebooks'])
        pure_markdowns = find_files(build['non-notebooks'], src_dir,
                                    build['exclusions'])
        depends = find_files(build['dependencies'], src_dir)
        return sorted(notebooks), sorted(pure_markdowns), sorted(depends)

    def _get_updated_md_files(self):
        notebooks, pure_markdowns, depends = self._find_md_files()
        depends_mtimes = get_mtimes(depends)
        latest_depend = max(depends_mtimes) if len(depends_mtimes) else 0
        updated_notebooks = get_updated_files(
            notebooks, self.config.src_dir, self.config.eval_dir, 'md', 'ipynb', latest_depend)
        updated_markdowns = get_updated_files(
            pure_markdowns, self.config.src_dir, self.config.eval_dir, 'md', 'md', latest_depend)
        return updated_notebooks, updated_markdowns

    def outputcheck(self):
        notebooks, _, _ = self._find_md_files()
        reader = notedown.MarkdownReader()
        error = False
        for fn in notebooks:
            with open(fn, 'r') as f:
                notebook= reader.read(f)
            for c in notebook.cells:
                if 'outputs' in c and len(c['outputs']):
                    logging.error("Found execution outputs in %s", fn)
                    error = True
        if error:
            exit(-1)

    @_once
    def eval(self):
        """Evaluate the notebooks and save them in a different folder"""
        eval_tik = datetime.datetime.now()
        notebooks, pure_markdowns, depends = self._find_md_files()
        depends_mtimes = get_mtimes(depends)
        latest_depend = max(depends_mtimes) if len(depends_mtimes) else 0
        updated_notebooks = get_updated_files(
            notebooks, self.config.src_dir, self.config.eval_dir, 'md', 'ipynb', latest_depend)
        updated_markdowns = get_updated_files(
            pure_markdowns, self.config.src_dir, self.config.eval_dir, 'md', 'md', latest_depend)
        num_updated_notebooks = len(updated_notebooks)
        num_updated_markdowns = len(updated_markdowns)
        logging.info('%d notebooks are outdated', num_updated_notebooks)
        for i, nb in enumerate(updated_notebooks):
            logging.info('[%d] %s', i + 1, nb[0])
        self._copy_resources(self.config.src_dir, self.config.eval_dir)

        for i, (src, tgt) in enumerate(updated_notebooks):
            tik = datetime.datetime.now()
            logging.info('[%d/%d, %s] Evaluating %s, save as %s',
                         i+1, num_updated_notebooks,
                         get_time_diff(eval_tik, tik), src, tgt)
            mkdir(os.path.dirname(tgt))
            run_cells = self.config.build['eval_notebook'].lower()
            process_and_eval_notebook(src, tgt, run_cells=='true')
            tok = datetime.datetime.now()

        for src, tgt in updated_markdowns:
            logging.info('Copying %s to %s', src, tgt)
            mkdir(os.path.dirname(tgt))
            shutil.copyfile(src, tgt)
        self._rm_tgt_files('md', 'ipynb', self.config.eval_dir)

    # Remove target files (e.g., eval and rst) based on removed files under src
    def _rm_tgt_files(self, src_ext, tgt_ext, tgt_dir, must_incls=None):
        notebooks_to_rm = get_files_to_rm(
                self.config.build['notebooks'], self.config.src_dir, tgt_dir,
                src_ext, tgt_ext)

        non_notebooks_pattern = (self.config.build['non-notebooks'] + ' '
                + self.config.build['resources'])
        non_notebooks_to_rm = get_files_to_rm(non_notebooks_pattern,
                self.config.src_dir, tgt_dir)

        if must_incls:
            must_incls = set(must_incls)
        tgt_files_to_rm = [f for f in notebooks_to_rm + non_notebooks_to_rm
                           if not must_incls or f not in must_incls]
        if tgt_files_to_rm:
            logging.info('Cleaning target files whose corresponding source '
                'files are removed %s', ','.join(tgt_files_to_rm))
            for fn in tgt_files_to_rm:
                os.remove(fn)
            rm_empty_dir(tgt_dir)

    def _copy_resources(self, src_dir, tgt_dir):
        resources = self.config.build['resources']
        if resources:
            logging.info('Copying resources "%s" from %s to %s',
                         ' '.join(resources.split()), src_dir, tgt_dir)
        for res in resources.split():
            src = os.path.join(src_dir, res)
            updated = get_updated_files(find_files(src), src_dir, tgt_dir)
            for src, tgt in updated:
                if os.path.isdir(src):
                    continue
                mkdir(os.path.dirname(tgt))
                shutil.copyfile(src, tgt)

    def _copy_rst(self):
        rst_files = find_files(self.config.build['rsts'], self.config.src_dir)
        updated_rst = get_updated_files(rst_files, self.config.src_dir, self.config.rst_dir)
        if len(updated_rst):
            logging.info('Copy %d updated RST files to %s',
                         len(updated_rst), self.config.rst_dir)
        for src, tgt in updated_rst:
            copy(src, tgt)
        return rst_files

    @_once
    def rst(self):
        self.eval()
        notebooks = find_files(os.path.join(self.config.eval_dir, '**', '*.ipynb'))
        updated_notebooks = get_updated_files(
            notebooks, self.config.eval_dir, self.config.rst_dir, 'ipynb', 'rst')
        logging.info('%d rst files are outdated', len(updated_notebooks))
        for src, tgt in updated_notebooks:
            logging.info('Convert %s to %s', src, tgt)
            mkdir(os.path.dirname(tgt))
            ipynb2rst(src, tgt)
        # Generate conf.py under rst folder
        prepare_sphinx_env(self.config)
        self._copy_rst()
        self._copy_resources(self.config.src_dir, self.config.rst_dir)

        must_incl_rst_files = get_tgt_files_from_src_pattern(
                self.config.build['rsts'], self.config.rst_dir, 'rst', 'rst')
        self._rm_tgt_files('md', 'rst', self.config.rst_dir,
                           must_incl_rst_files)

    @_once
    def html(self):
        self.rst()
        self.colab()
        self.sagemaker()
        run_cmd(['sphinx-build', self.config.rst_dir, self.config.html_dir,
                 '-b html -c', self.config.rst_dir, self.sphinx_opts])
        colab.add_button(self.config.colab, self.config.html_dir)

    @_once
    def ipynb(self):
        self.eval()
        run_cmd(['rm -rf', self.config.ipynb_dir, '; cp -r ',
                 self.config.eval_dir, self.config.ipynb_dir])
        update_ipynb_toc(self.config.ipynb_dir)

    @_once
    def colab(self):
        self.ipynb()
        colab.generate_notebooks(
            self.config.colab, self.config.ipynb_dir, self.config.colab_dir)

    @_once
    def sagemaker(self):
        self.ipynb()
        sagemaker.generate_notebooks(
            self.config.sagemaker, self.config.ipynb_dir, self.config.sagemaker_dir)

    @_once
    def linkcheck(self):
        self.rst()
        run_cmd(['sphinx-build', self.config.rst_dir, self.config.linkcheck_dir,
                 '-b linkcheck -c', self.config.rst_dir, self.sphinx_opts])

    @_once
    def pdf(self):
        self.rst()
        run_cmd(['sphinx-build ', self.config.rst_dir, self.config.pdf_dir,
                 '-b latex -c', self.config.rst_dir, self.sphinx_opts])

        script = self.config.pdf['post_latex']
        process_latex(self.config.tex_fname, script)
        run_cmd(['cd', self.config.pdf_dir, '&& make'])

    @_once
    def pkg(self):
        self.ipynb()
        zip_fname = 'out.zip'
        run_cmd(['cd', self.config.ipynb_dir, '&& zip -r',
                 zip_fname, '*'])
        shutil.move(os.path.join(self.config.ipynb_dir, zip_fname),
                    self.config.pkg_fname)

    @_once
    def lib(self):
        save_mark = self.config.library['save_mark']
        if not save_mark:
            logging.info('No save mark is specified, ignoring...')
            return
        lib_fname = self.config.library['save_filename']
        logging.info('Matching with the partten: "%s"', save_mark)

        root = os.path.join(self.config.src_dir, self.config.build['index'] + '.md')
        notebooks = get_toc(root)
        notebooks_enabled, _, _ = self._find_md_files()
        notebooks = [nb for nb in notebooks if nb in notebooks_enabled]
        with open(lib_fname, 'w') as f:
            lib_name = os.path.dirname(lib_fname)
            assert not '/' in lib_name, lib_name
            f.write('# This file is generated automatically through:\n')
            f.write('#    d2lbook build lib\n')
            f.write('# Don\'t edit it directly\n\n')
            f.write('import sys\n'+lib_name+' = sys.modules[__name__]\n\n')

            for nb in notebooks:
                blocks = get_code_to_save(nb, save_mark)
                if blocks:
                    logging.info('Found %d blocks in %s', len(blocks), nb)
                    for block in blocks:
                        logging.info(' --- %s', block[0])
                        code = '# Defined in file: %s\n%s\n\n\n' %(
                            nb, '\n'.join(block))
                        f.write(code)

        logging.info('Saved into %s', lib_fname)

    def all(self):
        self.eval()
        self.rst()
        self.html()
        self.pdf()
        self.pkg()

def get_code_to_save(input_fn, save_mark):
    """get the code blocks (import, class, def) that will be saved"""
    reader = notedown.MarkdownReader(match='strict')
    with open(input_fn, 'r') as f:
        nb = reader.read(f)
    saved = []
    for cell in nb.cells:
        if cell.cell_type == 'code':
            lines = cell.source.split('\n')
            for i, l in enumerate(lines):
                if l.strip().startswith('#') and save_mark in l:
                    block = [lines[i+1]]
                    # For code block only containing import statements (e.g., in
                    # preface.md)
                    if lines[i+1].startswith('import') or lines[i+1].startswith('from'):
                        for j in range(i+2, len(lines)):
                            block.append(lines[j])
                    # For code blocks containing def or class
                    else:
                        for j in range(i+2, len(lines)):
                            l = lines[j]
                            if not l.startswith(' ') and len(l):
                                break
                            block.append(l)
                    if len(block[-1]) == 0:
                        del block[-1]
                    saved.append(block)
    return saved

def split_markdown(source):
    """Split markdown into a list of text and code blocks"""
    cells = []
    in_code = False
    cur_code_mark = None
    cur_code_class = None
    cur_src = []
    code_mark = re.compile('(```+) *(.*)')
    def _add_cell():
        if cur_src:
            if in_code:
                cells.append({'type':'code', 'class':cur_code_class,
                            'source':'\n'.join(cur_src)})
            else:
                cells.append({'type':'markdown', 'source':'\n'.join(cur_src)})
    for l in source.split('\n'):
        ret = code_mark.match(l)
        if ret:
            if not in_code:
                _add_cell()
                cur_code_mark, cur_code_class = ret.groups()
                cur_src = []
                in_code = True
            else:
                if ret.groups()[0] == cur_code_mark:
                    _add_cell()
                    cur_src = []
                    in_code = False
                else:
                    cur_src.append(l)
        else:
            cur_src.append(l)
    _add_cell()
    return cells

def join_markdown_cells(cells):
    src = []
    for c in cells:
        if c['type'] == 'markdown':
            src.append(c['source'])
        else:
            src += ['```'+c['class'], c['source'], '```']
    return '\n'.join(src)

def update_ipynb_toc(root):
    """Change the toc code block into a list of clickable links"""
    notebooks = find_files('**/*.ipynb', root)
    for fn in notebooks:
        with open(fn, 'r') as f:
            notebook = nbformat.read(f, as_version=4)
        for cell in notebook.cells:
            if (cell.cell_type == 'markdown' and '```toc' in cell.source):
                md_cells = split_markdown(cell.source)
                for c in md_cells:
                    if c['type'] == 'code' and c['class'] == 'toc':
                        toc = []
                        for l in c['source'].split('\n'):
                            if l and not l.startswith(':'):
                                toc.append(' - [%s](%s.ipynb)'%(l,l))
                        c['source'] = '\n'.join(toc)
                        c['type'] = 'markdown'
                cell.source = join_markdown_cells(md_cells)
        with open(fn, 'w') as f:
            f.write(nbformat.writes(notebook))

def get_toc(root):
    """return a list of files in the order defined by TOC"""
    subpages = get_subpages(root)
    res = [root]
    for fn in subpages:
        res.extend(get_toc(fn))
    return res

def get_subpages(input_fn):
    """read toc in input_fn, returns what it contains"""
    subpages = []
    reader = notedown.MarkdownReader()
    with open(input_fn, 'r') as f:
        nb = reader.read(f)
    for cell in nb.cells:
        if (cell.cell_type == 'code' and
            'attributes' in cell.metadata and
            'toc' in cell.metadata.attributes['classes']):
            for l in cell.source.split('\n'):
                l = l.strip()
                if not l.startswith(':'):
                    fn = os.path.join(os.path.dirname(input_fn), l + '.md')
                    if os.path.exists(fn):
                        subpages.append(fn)
    return subpages

def process_and_eval_notebook(input_fn, output_fn, run_cells, timeout=20*60,
                              lang='python'):
    # process: add empty lines before and after a mark, otherwise it confuses
    # the rst parser
    with open(input_fn, 'r') as f:
        md = f.read()
    lines = md.split('\n')
    in_code = CharInMDCode(lines)
    for i, line in enumerate(lines):
        m = mark_re_md.match(line)
        if (m is not None
            and m[1] not in ('ref', 'numref', 'eqref')
            and not in_code.in_code(i,0)
            and m.end() == len(line)):
            lines[i] = '\n'+line+'\n'
    reader = notedown.MarkdownReader(match='strict')
    notebook = reader.reads('\n'.join(lines))
    # evaluate
    if run_cells:
        # change to the notebook directory to resolve the relpaths properly
        cwd = os.getcwd()
        os.chdir(os.path.join(cwd, os.path.dirname(output_fn)))
        notedown.run(notebook, timeout)
        os.chdir(cwd)
    # write
    notebook['metadata'].update({'language_info':{'name':lang}})
    with open(output_fn, 'w') as f:
        f.write(nbformat.writes(notebook))

def delete_lines(lines, deletes):
    return [line for i, line in enumerate(lines) if i not in deletes]

class CharInMDCode(object):
    """
    Indicates if a char (at line_i, pos of in_code) is in a code block of md.

    Examples:

    i) The following chars (including lines starting with ```) are all chars in
       a code block of md.

       ```bash
       pip install d2lbook
       ```

    ii) Chars `d2lbook` (including ``) in the following sentence are all in a
        code block of md:

        Let us install `d2lbook` first.
    """
    def __init__(self, lines):
        in_code = []
        code_block_mark = None
        for line in lines:
            if not code_block_mark:
                if self._get_code_block_mark(line):
                    code_block_mark = self._get_code_block_mark(line)
                    in_code.append([True]*len(line))
                else:
                    char_in_code = False
                    code_line = [False] * len(line)
                    for i, char in enumerate(line):
                        if char == '`':
                            code_line[i] = True
                            char_in_code ^= True
                        elif char_in_code:
                            code_line[i] = True
                    in_code.append(code_line)
            else:
                in_code.append([True] * len(line))
                if line.strip().startswith(code_block_mark):
                    code_block_mark = None
        self._in_code = in_code

    def _match_back_quote(self, line):
        mark = ''
        for char in line:
            if char == '`':
                mark += '`'
            else:
                break
        return mark

    def _get_code_block_mark(self, line):
        ls = line.strip()
        if ls.startswith('```'):
            return self._match_back_quote(ls)
        return None

    def in_code(self, line_i, pos):
        return self._in_code[line_i][pos]

def process_rst(body):
    def indented(line):
        return line.startswith('   ')

    def blank(line):
        return len(line.strip()) == 0

    def look_behind(i, cond, lines):
        indices = []
        while i < len(lines) and cond(lines[i]):
            indices.append(i)
            i = i + 1
        return indices

    lines = body.split('\n')
    # deletes: indices of lines to be deleted
    i, deletes = 0, []
    while i < len(lines):
        line = lines[i]
        # '.. code:: toc' -> '.. toctree::', then remove consecutive empty lines
        # after the current line
        if line.startswith('.. code:: toc'):
            # convert into rst's toc block
            lines[i] = '.. toctree::'
            blanks = look_behind(i+1, blank, lines)
            deletes.extend(blanks)
            i += len(blanks)
        # .. code:: eval_rst
        #
        #
        #    .. only:: html
        #
        #       References
        #       ==========
        # ->
        #
        #
        #
        # .. only:: html
        #
        #    References
        #    ==========
        elif line.startswith('.. code:: eval_rst'):
            # make it a rst block
            deletes.append(i)
            j = i + 1
            while j < len(lines):
                line_j = lines[j]
                if indented(line_j):
                    lines[j] = line_j[3:]
                elif not blank(line_j):
                    break
                j += 1
            i = j
        elif line.startswith('.. parsed-literal::'):
            # add a output class so we can add customized css
            lines[i] += '\n    :class: output'
            i += 1
        # .. figure:: ../img/jupyter.png
        #    :alt: Output after running Jupyter Notebook. The last row is the URL
        #    for port 8888.
        #
        #    Output after running Jupyter Notebook. The last row is the URL for
        #    port 8888.
        #
        # :width:``700px``
        #
        # :label:``fig_jupyter``
        #->
        # .. _fig_jupyter:
        #
        # .. figure:: ../img/jupyter.png
        #    :width: 700px
        #
        #    Output after running Jupyter Notebook. The last row is the URL for
        #    port 8888.
        elif indented(line) and ':alt:' in line:
            # Image caption, remove :alt: block, it cause trouble for long captions
            caps = look_behind(i, lambda l: indented(l) and not blank(l), lines)
            deletes.extend(caps)
            i += len(caps)
        # .. table:: Dataset versus computer memory and computational power
        #    +-...
        #    |
        #    +-...
        #
        # :label:``tab_intro_decade``
        # ->
        # .. _tab_intro_decade:
        #
        # .. table:: Dataset versus computer memory and computational power
        #
        #    +-...
        #    |
        #    +-...
        #
        elif line.startswith('.. table::'):
            # Add indent to table caption for long captions
            caps = look_behind(i+1, lambda l: not indented(l) and not blank(l),
                               lines)
            for j in caps:
                lines[j] = '   ' + lines[j]
            i += len(caps) + 1
        else:
            i += 1

    # change :label:my_label: into rst format
    lines = delete_lines(lines, deletes)
    deletes = []

    for i, line in enumerate(lines):
        pos, new_line = 0, ''
        while True:
            match = mark_re.search(line, pos)
            if match is None:
                new_line += line[pos:]
                break
            start, end = match.start(), match.end()
            # e.g., origin=':label:``fig_jupyter``', key='label', value='fig_jupyter'
            origin, key, value = match[0], match[1], match[2]
            new_line += line[pos:start]
            pos = end

            # assert key in ['label', 'eqlabel', 'ref', 'numref', 'eqref', 'width', 'height'], 'unknown key: ' + key
            if key == 'label':
                new_line += '.. _' + value + ':'
            elif key in ['ref', 'numref', 'cite']:
                new_line += ':'+key+':`'+value+'`'
            elif key == 'eqref':
                new_line += ':eq:`'+value+'`'
            # .. math:: f
            #
            # :eqlabel:``gd-taylor``
            # ->
            # .. math:: f
            #    :label: gd-taylor
            elif key == 'eqlabel':
                new_line += '   :label: '+value
                if blank(lines[i-1]):
                    deletes.append(i-1)
            elif key in ['width', 'height']:
                new_line += '   :'+key+': '+value
            elif key == 'bibliography':
                # a hard coded plain bibtex style...
                new_line += ('.. bibliography:: ' + value +
                             '\n   :style: apa')
                             # '\n   :style: apa\n   :all:') MM 20200104 removed ':all:' so only the cited references get printed
            else:
                logging.fatal('unknown key', key)

        lines[i] = new_line
    lines = delete_lines(lines, deletes)

    def move(i, j): # move line i to line j
        lines.insert(j, lines[i])
        if i > j:
            del lines[i+1]
        else:
            del lines[i]

    # move :width: or :width: just below .. figure::
    for i, line in enumerate(lines):
        if line.startswith('.. figure::'):
            for j in range(i+1, len(lines)):
                line_j = lines[j]
                if not indented(line_j) and not blank(line_j):
                    break
                if line_j.startswith('   :width:') or line_j.startswith('   :height:'):
                    move(j, i+1)

    # move .. _label: before a image, a section, or a table
    lines.insert(0, '')
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('.. _'):
            for j in range(i-1, -1, -1):
                line_j = lines[j]
                if (line_j.startswith('.. table:')
                    or line_j.startswith('.. figure:')):
                    move(i, j-1)
                    lines.insert(j-1, '')
                    i += 1  # Due to insertion of a blank line
                    break
                if (len(set(line_j)) == 1
                    and line_j[0] in ['=','~','_', '-']):
                    k = max(j-2, 0)
                    move(i, k)
                    lines.insert(k, '')
                    i += 1  # Due to insertion of a blank line
                    break
        i += 1

    # change .. image:: to .. figure:: to they will be center aligned
    for i, line in enumerate(lines):
        if '.. image::' in line:
            lines[i] = line.replace('.. image::', '.. figure::')

    # sometimes the code results contains vt100 codes, widely used for
    # coloring, while it is not supported by latex.
    for i, l in enumerate(lines):
        lines[i] = re.sub(r'\x1b\[[\d;]*K', '',
                          re.sub(r'\x1b\[[\d;]*m', '', l))

    return '\n'.join(lines)

def ipynb2rst(input_fn, output_fn):
    with open(input_fn, 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            if '# hide outputs' in cell.source.lower():
                cell.outputs = []
            if '# hide code' in cell.source.lower():
                cell.source = ''

    writer = nbconvert.RSTExporter()
    sig = hashlib.sha1(input_fn.encode()).hexdigest()[:6]
    resources = {'unique_key':
                 'output_'+rm_ext(os.path.basename(output_fn))+'_'+sig}
    (body, resources) = writer.from_notebook_node(notebook, resources)

    # Process the raw rst file generated by nbconvert to output a new rst file
    body = process_rst(body)

    with open(output_fn, 'w') as f:
        f.write(body)
    outputs = resources['outputs']
    base_dir = os.path.dirname(output_fn)
    for fn in outputs:
        full_fn = os.path.join(base_dir, fn)
        with open(full_fn, 'wb') as f:
            f.write(outputs[fn])

def process_latex(fname, script):
    with open(fname, 'r') as f:
        lines = f.read().split('\n')

    _combine_citations(lines)
    _center_graphics(lines)

    with open(fname, 'w') as f:
        f.write('\n'.join(lines))
    # Execute custom process_latex script
    if script:
        cmd = "python " + script + " " + fname
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logging.error('%s', stderr.decode())
            exit(-1)

def _combine_citations(lines):
    # convert \sphinxcite{A}\sphinxcite{B} to \sphinxcite{A,B}
    for i, l in enumerate(lines):
        if '}\sphinxcite{' in l:
            lines[i] = l.replace('}\sphinxcite{', ',')

# E.g., tag = 'begin{figure}'
def _tag_in_line(tag, line):
    assert '\\' not in set(tag)
    return any([elem.startswith(tag) for elem in line.split('\\')])

def _center_graphics(lines):
    tabulary_cnt = 0
    figure_cnt = 0
    in_doc = False
    for i, l in enumerate(lines):
        if _tag_in_line('begin{tabulary}', l):
            tabulary_cnt += 1
        elif _tag_in_line('end{tabulary}', l):
            tabulary_cnt -= 1
        elif _tag_in_line('begin{figure}', l):
            figure_cnt += 1
        elif _tag_in_line('end{figure}', l):
            figure_cnt -= 1
        elif _tag_in_line('begin{document}', l):
            in_doc = True

        # 'tabulary' and 'figure' blocks do include '\centering', so only center
        # '\sphinxincludegraphics{}' by enclosing it with '\begin{center}'
        # '\end{center}'. Logo should not be centered and it is not in_doc.
        if tabulary_cnt == 0 and figure_cnt == 0 and in_doc:
            sigs_greedy = re.findall('\\\\sphinxincludegraphics\\{.*\\}', l)
            if len(sigs_greedy) > 0:
                longest_balanced_braces = regex.findall('\{(?>[^{}]|(?R))*\}',
                                                        sigs_greedy[0])
                sig_with_balanced_braces = ('\\sphinxincludegraphics'
                                            + longest_balanced_braces[0])
                lines[i] = l.replace(sig_with_balanced_braces,
                                     ('\\begin{center}'
                                      + sig_with_balanced_braces
                                      + '\\end{center}'))
