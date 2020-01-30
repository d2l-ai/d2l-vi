# Building this Website

You may find building this website is a good starting point for your
project. The source codes of this site is available under
[demo/](https://github.com/d2l-ai/d2l-book/tree/master/demo).

Please make sure you have `git` (e.g. `conda install git`),  `numpy` and
`matplotlib` (e.g. `pip install numpy matplotlib`) installed.
The following command will download the source codes, evaluate all notebooks and generate outputs in
`ipynb`, `html` and `pdf` format.

```sh
git clone https://github.com/d2l-ai/d2l-book
cd d2l-book/demo
d2lbook build all
```

Once finished, you can check the results in the `_build` folder. For example, this page is in `_build/html/index.html`, the PDF version is at `_build/pdf/d2l-book.pdf`, all evaluated notebooks are under `_build/eval/`.

You can build a particular format:

```sh
d2lbook build eval  # evaluate noteboks and save in .ipynb formats
d2lbook build html  # build the HTML version
d2lbook build pdf   # build the PDF version
```
