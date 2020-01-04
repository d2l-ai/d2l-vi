alias d2l_build='docker build -t d2l . --no-cache'
# to specify the current directory (inside mly project folder):
# use `pwd` if you are in unix;
# use %CD% if you are in windows
# reference: https://github.com/moby/moby/issues/4830#issuecomment-264366876
alias d2l_run='docker run -i -t -v `pwd`:/d2l d2l'
