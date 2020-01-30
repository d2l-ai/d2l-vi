# Building 

This section we will explain various options to build your projects. This options can be grouped into four categories:

1. Sanity check
   - `d2lbook build linkcheck` will check if all internal and external links are accessible.  
   - `d2lbook build outputcheck` will check if no notebook will contain code outputs
1. Building results
   - `d2lbook build html`: build the HTML version into `_build/html`
   - `d2lbook build pdf`: build the PDF version into `_build/pdf`
   - `d2lbook build pkg`: build a zip file contains all `.ipynb` notebooks
1. Additional features   
   - `d2lbook build colab`: convert all notebooks can be run on Google Colab into `_build/colab`. See more in :numref:`sec_colab`
   - `d2lbook build lib`: build a Python package so we can reuse codes in other notebooks. See more in XXX.
1. Internal stages, which often are triggered automatically.  
   - `d2lbook build eval`: evaluate all notebooks and save them as `.ipynb` notebooks into `_build/eval`
   - `d2lbook build rst`: convert all notebooks into `rst` files and create a Sphinx project in `_build/rst`
   

## Building Cache

We encourage you to evaluate your notebooks to obtain code cell results, instead of keeping these results in the source files for two reasons:
1. These results make code review difficult, especially when they have randomness either due to numerical precision or random number generators. 
1. A notebook hasn't evaluated for a while may be broken due to package upgrading. 

But the evaluation costs additional overhead during building. We recommend to limit the runtime for each notebook within a few minutes. And `d2lbook` will reuse the previous built and only evaluate the modified notebooks.

For example, the average runtime of a notebook (section) in [Dive into Deep Learning](https://d2l.ai) is about 2 minutes on a GPU machine, due to training neural networks. It contains more than 100 notebooks, which make the total runtime cost 2-3 hours. In reality, each code change will only modify a few notebooks and therefore the [build time](http://ci.d2l.ai/blue/organizations/jenkins/d2l-en/activity) is often less than 10 minutes. 

Let's see how it works. First create a project as we did in :numref:`sec_create`. 

```{.python .input}
!mkdir -p cache
```

```{.python .input}
%%writefile cache/index.md
# My Book

The starting page of my book with `d2lbook`.

````toc
get_started
````
```

```{.python .input}
%%writefile cache/get_started.md
# Getting Started

Please first install my favorite package `numpy`.
```

```{.python .input}
!cd cache; d2lbook build html
```

You can see `index.md` is evaluated. (Though it doesn't contain codes, it's fine to evaluate it as a Jupyter notebook.)

If building again, we will see no notebook will be evaluated.

```{.python .input}
!cd cache; d2lbook build html
```

Now let's modify `get_started.md`, you will see it will be re-evaluated, but not `index.md`.  

```{.python .input}
%%writefile cache/get_started.md
# Getting Started

Please first install my favorite package `numpy>=1.18`.
```

```{.python .input}
!cd cache; d2lbook build html
```

One way to trigger the whole built is removing the saved notebooks in `_build/eval`, or simply deleting `_build`. Another way is specifying some dependencies. For example, in the following cell we add `config.ini` into the dependencies. Every time `config.ini` is modified, it will invalid the cache of all notebooks and trigger a build from scratch. 


```{.python .input}
%%writefile cache/config.ini

[build]
dependencies = config.ini
```

```{.python .input}
!cd cache; d2lbook build html
```

Last, let's clean our workspace. 

```{.python .input}
!rm -rf cache
```
