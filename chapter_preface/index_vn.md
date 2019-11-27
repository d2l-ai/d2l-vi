<!--
# Preface
-->

# *translate the above header*

<!--
Just a few years ago, there were no legions of deep learning scientists
developing intelligent products and services at major companies and startups.
When the youngest among us (the authors) entered the field,
machine learning did not command headlines in daily newspapers.
Our parents had no idea what machine learning was,
let alone why we might prefer it to a career in medicine or law.
Machine learning was a forward-looking academic discipline
with a narrow set of real-world applications.
And those applications, e.g., speech recognition and computer vision,
required so much domain knowledge that they were often regarded
as separate areas entirely for which machine learning was one small component.
Neural networks then, the antecedents of the deep learning models
that we focus on in this book, were regarded as outmoded tools.
-->

*translate the above block*

<!--
In just the past five years, deep learning has taken the world by surprise,
driving rapid progress in fields as diverse as computer vision,
natural language processing, automatic speech recognition,
reinforcement learning, and statistical modeling.
With these advances in hand, we can now build cars that drive themselves
with more autonomy than ever before (and less autonomy 
than some companies might have you believe), 
smart reply systems that automatically draft the most mundane emails,
helping people dig out from oppressively large inboxes,
and software agents that dominate the world's best humans 
at board games like Go, a feat once thought to be decades away.
Already, these tools exert ever-wider impacts on industry and society,
changing the way movies are made, diseases are diagnosed,
and playing a growing role in basic sciences---from astrophysics to biology.
-->

*translate the above block*

<!--
## About This Book
-->

## *translate the above header*

<!--
This book represents our attempt to make deep learning approachable,
teaching you both the *concepts*, the *context*, and the *code*.
-->

*translate the above block*

<!--
### One Medium Combining Code, Math, and HTML
-->

### *translate the above header*

<!--
For any computing technology to reach its full impact,
it must be well-understood, well-documented, and supported by
mature, well-maintained tools.
The key ideas should be clearly distilled,
minimizing the onboarding time needing to bring new practitioners up to date.
Mature libraries should automate common tasks,
and exemplar code should make it easy for practitioners
to modify, apply, and extend common applications to suit their needs.
Take dynamic web applications as an example.
Despite a large number of companies, like Amazon,
developing successful database-driven web applications in the 1990s,
the potential of this technology to aid creative entrepreneurs
has been realized to a far greater degree in the past ten years,
owing in part to the development of powerful, well-documented frameworks.
-->

*translate the above block*

<!--
Testing the potential of deep learning presents unique challenges 
because any single application brings together various disciplines.
Applying deep learning requires simultaneously understanding
(i) the motivations for casting a problem in a particular way;
(ii) the mathematics of a given modeling approach;
(iii) the optimization algorithms for fitting the models to data;
and (iv) and the engineering required to train models efficiently,
navigating the pitfalls of numerical computing 
and getting the most out of available hardware.
Teaching both the critical thinking skills required to formulate problems,
the mathematics to solve them, and the software tools to implement those
solutions all in one place presents formidable challenges.
Our goal in this book is to present a unified resource
to bring would-be practitioners up to speed.
-->

*translate the above block*

<!--
We started this book project in July 2017 when we needed 
to explain MXNet's (then new) Gluon interface to our users.
At the time, there were no resources that simultaneously
(i) were up to date; (ii) covered the full breadth 
of modern machine learning with substantial technical depth;
and (iii) interleaved exposition of the quality one expects 
from an engaging textbook with the clean runnable code 
that one expects to find in hands-on tutorials.
We found plenty of code examples for 
how to use a given deep learning framework
(e.g., how to do basic numerical computing with matrices in TensorFlow)
or for implementing particular techniques 
(e.g., code snippets for LeNet, AlexNet, ResNets, etc)
scattered across various blog posts and GitHub repositories.
However, these examples typically focused on
*how* to implement a given approach,
but left out the discussion of *why* certain algorithmic decisions are made.
While some interactive resources have popped up sporadically 
to address a particular topic, e.g., the engaging blog posts
published on the website [Distill](http://distill.pub), or personal blogs,
they only covered selected topics in deep learning, 
and often lacked associated code.
On the other hand, while several textbooks have emerged,
most notably :cite:`Goodfellow.Bengio.Courville.2016`,
which offers a comprehensive survey of the concepts behind deep learning,
these resources do not marry the descriptions 
to realizations of the concepts in code,
sometimes leaving readers clueless as to how to implement them.
Moreover, too many resources are hidden behind the paywalls 
of commercial course providers.
-->

*translate the above block*

<!--
We set out to create a resource that could
(1) be freely available for everyone;
(2) offer sufficient technical depth to provide a starting point on the path
to actually becoming an applied machine learning scientist;
(3) include runnable code, showing readers *how* to solve problems in practice;
(4) that allowed for rapid updates, both by us 
and also by the community at large;
and (5) be complemented by a [forum](http://discuss.mxnet.io)
for interactive discussion of technical details and to answer questions.
-->

*translate the above block*

<!--
These goals were often in conflict.
Equations, theorems, and citations are best managed and laid out in LaTeX.
Code is best described in Python.
And webpages are native in HTML and JavaScript.
Furthermore, we want the content to be
accessible both as executable code, as a physical book,
as a downloadable PDF, and on the internet as a website.
At present there exist no tools and no workflow
perfectly suited to these demands, so we had to assemble our own.
We describe our approach in detail in :numref:`sec_how_to_contribute`.
We settled on Github to share the source and to allow for edits,
Jupyter notebooks for mixing code, equations and text,
Sphinx as a rendering engine to generate multiple outputs,
and Discourse for the forum.
While our system is not yet perfect, 
these choices provide a good compromise among the competing concerns.
We believe that this might be the first book published 
using such an integrated workflow.
-->

*translate the above block*

<!--
### Learning by Doing
-->

### *translate the above header*

<!--
Many textbooks teach a series of topics, each in exhaustive detail.
For example, Chris Bishop's excellent textbook :cite:`Bishop.2006`,
teaches each topic so thoroughly, that getting to the chapter
on linear regression requires a non-trivial amount of work.
While experts love this book precisely for its thoroughness,
for beginners, this property limits its usefulness as an introductory text.
-->

*translate the above block*

<!--
In this book, we will teach most concepts *just in time*.
In other words, you will learn concepts at the very moment
that they are needed to accomplish some practical end.
While we take some time at the outset to teach
fundamental preliminaries, like linear algebra and probability,
we want you to taste the satisfaction of training your first model
before worrying about more esoteric probability distributions.
-->

*translate the above block*

<!--
Aside from a few preliminary notebooks that provide a crash course
in the basic mathematical background,
each subsequent chapter introduces both a reasonable number of new concepts
and provides single self-contained working examples---using real datasets.
This presents an organizational challenge.
Some models might logically be grouped together in a single notebook.
And some ideas might be best taught by executing several models in succession.
On the other hand, there is a big advantage to adhering
to a policy of *1 working example, 1 notebook*:
This makes it as easy as possible for you to
start your own research projects by leveraging our code.
Just copy a notebook and start modifying it.
-->

*translate the above block*

<!--
We will interleave the runnable code with background material as needed.
In general, we will often err on the side of making tools
available before explaining them fully (and we will follow up by
explaining the background later).
For instance, we might use *stochastic gradient descent*
before fully explaining why it is useful or why it works.
This helps to give practitioners the necessary
ammunition to solve problems quickly,
at the expense of requiring the reader 
to trust us with some curatorial decisions.
-->

*translate the above block*

<!--
Throughout, we will be working with the MXNet library,
which has the rare property of being flexible enough for research
while being fast enough for production.
This book will teach deep learning concepts from scratch.
Sometimes, we want to delve into fine details about the models
that would typically be hidden from the user 
by Gluon's advanced abstractions.
This comes up especially in the basic tutorials,
where we want you to understand everything 
that happens in a given layer or optimizer.
In these cases, we will often present two versions of the example:
one where we implement everything from scratch,
relying only on the NumPy interface and automatic differentiation,
and another, more practical example, 
where we write succinct code using Gluon.
Once we have taught you how some component works,
we can just use the Gluon version in subsequent tutorials.
-->

*translate the above block*

<!--
### Content and Structure
-->

### *translate the above header*

<!--
The book can be roughly divided into three parts, 
which are presented by different colors in :numref:`fig_book_org`:
-->

*translate the above block*

<!--
![Book structure](../img/book-org.svg)
