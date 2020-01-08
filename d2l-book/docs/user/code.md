# Code Cells
:label:`sec_code`

## Maximum Line Length

We recommend you to set the maximum line length to be 78 to avoid automatic line break in PDF. You can enable the Ruler extension in [nbextensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions) to add visual vertical line in Jupyter when writing codes. 

```{.python .input}
'-' * 78
```

## Hide Source and Outputs

We can hide the source of a code cell by adding a comment line `# Hide
code` in the cell. We can also hide the code cell outputs using `# Hide outputs`

For example, here is the normal code cell:

```{.python .input}
1+2+3
```

Let's hide the source codes

```{.python .input}
# Hide code
1+2+3
```

Also try hiding the outputs

```{.python .input}
# Hide outputs
1+2+3
```

## Plotting

We recommend you to use the `svg` format to plot a figure. For example, the following code configures `matplotlib`

```{.python .input  n=3}
%matplotlib inline
from IPython import display
from matplotlib import pyplot as plt
import numpy as np

display.set_matplotlib_formats('svg')

x = np.arange(0, 10, 0.1)
plt.plot(x, np.sin(x));
```
