# Installation

The `d2lbook` package is tested under macOS and Linux. (You are welcome to
contribute a Windows release).

First make sure you have [pip](https://pip.pypa.io/en/stable/) available. In
option, we recommend [conda](https://docs.conda.io/en/latest/miniconda.html) for
libraries that `pip` doesn't support.

Now install the command-line interface.

```sh
pip install git+https://github.com/d2l-ai/d2l-book
```

This is a [d2lbook pip package](https://pypi.org/project/d2lbook/), but we
recommend you to install latest version at Github directly since it's under fast
developing.

To build HTML results, we need [pandoc](https://pandoc.org/). You can install it
through `conda install pandoc`.

Building the PDF version requires
[LibRsvg](https://wiki.gnome.org/Projects/LibRsvg) to convert your SVG images
(our recommend format), e.g. `conda install librsvg`, and of course, you need to
have a LaTeX distribution, e.g. [Tex Live](https://www.tug.org/texlive/), available,
