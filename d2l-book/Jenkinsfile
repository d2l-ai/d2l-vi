stage("Build and Publish") {
  node {
    ws('workspace/d2l-book') {
      checkout scm
      sh '''set -ex
      conda remove -n d2l-book-build --all -y
      conda create -n d2l-book-build pip -y
      conda activate d2l-book-build
      pip install .
      cd docs
      pip install matplotlib numpy
      d2lbook build html pdf
      '''

      if (env.BRANCH_NAME == 'master') {
        sh '''set -ex
        conda activate d2l-book-build
        cd docs
        d2lbook deploy html pdf
        d2lbook clear
      '''
      }
    }
  }
}
