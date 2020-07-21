alias d2l_build='docker build -t d2l .'

# force bulding from scratch
# alias d2l_build='docker build -t d2l . --no-cache'
# force building from line after ARG
# alias d2l_build='docker build --build-arg D2L_VER=$(date +%Y%m%d-%H%M%S) -t d2l .'

# to specify the current directory (inside mly project folder):
# use `pwd` if you are in unix;
# use %CD% if you are in windows
# reference: https://github.com/moby/moby/issues/4830#issuecomment-264366876
alias d2l_run='docker run -i -t -v `pwd`:/d2l d2l'

alias d2l_build_run='docker build --build-arg D2L_VER=$(date +%Y%m%d-%H%M%S) -t d2l . && docker run -i -t -v `pwd`:/d2l d2l'


new_deploy () {
    export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
    d2lbook build html
    d2lbook deploy html
}
