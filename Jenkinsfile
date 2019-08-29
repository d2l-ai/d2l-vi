stage("Build and Publish") {
  node {
    ws('workspace/d2l-en') {
      checkout scm
<<<<<<< HEAD
      sh "git submodule update --init"
      sh "build/utils/sanity_check.sh"
      sh "build/utils/clean_build.sh"
      sh "conda env update -f build/env.yml"
      sh "build/utils/build_html.sh en"
      sh "build/utils/build_pdf.sh en"
      sh "build/utils/build_pkg.sh en"
      if (env.BRANCH_NAME == 'master') {
        sh "build/utils/publish_website.sh en"
        withCredentials([usernamePassword(credentialsId: '13cb1009-3cda-49ef-9aaa-8a705fdaaeb7',
                         passwordVariable: 'GIT_PASSWORD', usernameVariable: 'GIT_USERNAME')]) {
          sh "build/publish_notebook.sh"
        }
      }
=======
      def ENV_NAME = "${TASK}-${EXECUTOR_NUMBER}";
      def EID = EXECUTOR_NUMBER.toInteger()
      def CUDA_VISIBLE_DEVICES=(EID*2).toString() + ',' + (EID*2+1).toString();

      sh label: "Build Environment", script: """set -ex
      rm -rf ~/miniconda3/envs/${ENV_NAME}
      conda create -n ${ENV_NAME} pip python=3.7.3 -y
      conda activate ${ENV_NAME}
      pip install mxnet-cu100==1.5.0
      pip install git+https://github.com/d2l-ai/d2l-book
      python setup.py develop
      pip list
      """

      sh label: "Check Execution Output", script: """set -ex
      conda activate ${ENV_NAME}
      d2lbook build outputcheck
      """

      sh label: "Execute Notebooks", script: """set -ex
      conda activate ${ENV_NAME}
      export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
      d2lbook build eval
      """

      sh label:"Build HTML", script:"""set -ex
      conda activate ${ENV_NAME}
      ./static/build_html.sh
      """

      sh label:"Build PDF", script:"""set -ex
      conda activate ${ENV_NAME}
      d2lbook build pdf
      """

      sh label:"Build Package", script:"""set -ex
      conda activate ${ENV_NAME}
      # don't pack downloaded data into the pkg
      mv _build/eval/data _build/data_tmp
      cp -r data _build/eval
      d2lbook build html pkg
      rm -rf _build/eval/data
      mv _build/data_tmp _build/eval/data
      """

>>>>>>> 41671d5... disable Jenkings and config
	}
  }
}
