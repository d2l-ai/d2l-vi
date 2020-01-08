# Creating Your Project
:label:`sec_create`

Let's start with a simple project from scratch.

## Project From Scratch

First make a folder for our project.

```{.python .input  n=1}
!mkdir -p mybook
```

Then create two pages. The `index.md` is the index page which contains the
table of contents (TOC), which includes the other page `get_started.md`. Note
that the TOC is defined in a code block with tag `toc`. If you are familiar with
Sphinx, you can find it's similar to the TOC definition in Sphinx. Please refer
to :numref:`sec_markdown` for more extensions `d2lbook` added to markdown. Also note we used the build-in magic `writefile` to save a code block into file provided by [Jupyter](https://ipython.readthedocs.io/en/stable/interactive/magics.html).

```{.python .input  n=2}
%%writefile mybook/index.md
# My Book

The starting page of my book with `d2lbook`.

````toc
get_started
````
```

```{.python .input  n=3}
%%writefile mybook/get_started.md
# Getting Started

Please first install my favorite package `numpy`.
```

Now let's build the HTML version.

```{.python .input  n=4}
!cd mybook && d2lbook build html
```

The HTML index page is then available at `mybook/_build/html/index.html`.

## Configuration

You can customize how results are built and published through `config.ini` on the root folder.

```{.python .input  n=5}
%%writefile mybook/config.ini

[project]
# Specify the PDF filename to mybook.pdf
name = mybook  
# Specify the authors names in PDF
author = Adam Smith, Alex Li  

[html]
# Add two links on the navbar. A link consists of three
# items: name, URL, and a fontawesome icon. Items are separated by commas.
header_links = PDF, https://book.d2l.ai/d2l-book.pdf, fas fa-file-pdf,
               Github, https://github.com/d2l-ai/d2l-book, fab fa-github
```

Let's clear and build again.

```{.python .input}
!cd mybook && rm -rf _build && d2lbook build html
```

If you open `index.html` again, you will see the two links on the navigation bar. 

Let build the PDF output, you will find `Output written on mybook.pdf (7 pages).` in the output logs. 

```{.python .input}
!cd mybook && d2lbook build pdf
```

We will cover more configuration options in the following sections. You can check [default_config.ini](https://github.com/d2l-ai/d2l-book/blob/master/d2lbook/config_default.ini) for all configuration options and their default values. Also check these examples `config.ini` in

- [This website](https://github.com/d2l-ai/d2l-book/blob/master/docs/config.ini)
- [Dive into Deep Learning](https://github.com/d2l-ai/d2l-en/blob/master/config.ini)

Last, let's clear our workspace.

```{.python .input}
!rm -rf mybook
```
