# Editing Source Files

No matter whether it is a pure text file or a Jupyter notebook, we recommend that you save it as a markdown file. If it is a notebook, you can clear output before saving to make code review and version control easier. 

You can use your favorite markdown editors, e.g. [Typora](https://www.typora.io/), to edit markdown files directly. We enhanced markdown to support additional feature such as image/table captions and references, please refer to :numref:`sec_markdown` for more details. For a notebook, a Jupyter source code block is placed in a markdown code block with a `{.python .input}` tag, for example,

````
```{.python .input}
print('this is a Jupyter code cell')
```
````

Another way we recommend is using Jupyter to edit markdown files directly, especially when they contain source code blocks. Jupyter's default file format is `ipynb`. We can use the `notedown` plugin to have Jupyter open and save markdown files. 

You can install this extension by 

```bash
pip install mu-notedown
```

(`mu-notedown` is a fork of [notedown](https://github.com/aaren/notedown) with several modifications. You may need to uninstall the original `notedown` first.)

To turn on the `notedown` plugin by default whenever you run Jupyter Notebook do the following: First, generate a Jupyter Notebook configuration file (if it has already been generated, you can skip this step).

```bash
jupyter notebook --generate-config
```


Then, add the following line to the end of the Jupyter Notebook configuration file (for Linux/macOS, usually in the path `~/.jupyter/jupyter_notebook_config.py`):

```bash
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```


Next restart your Jupyter, you should be able to open these markdowns in Jupyter as notebooks now.

![Use Jupyter to edit :numref:`sec_create`](../img/jupyter.png)
:width:`500px`
