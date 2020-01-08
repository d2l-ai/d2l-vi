import os
import sys
import logging
import argparse
import shutil
from d2lbook.utils import *
from d2lbook.config import Config

__all__  = ['deploy']

commands = ['html', 'pdf', 'pkg', 'colab', 'sagemaker', 'all']

def deploy():
    parser = argparse.ArgumentParser(description='Deploy documents')
    parser.add_argument('commands', nargs='+', choices=commands)
    args = parser.parse_args(sys.argv[2:])
    config = Config()
    if config.deploy['s3_bucket']:
        deployer = S3Deployer(config)
    elif config.deploy['github_repo']:
        deployer = GithubDeployer(config)
    else:
        logging.fatal('No deployment URL. You need to specify either'
                      'a Github repo or a S3 bucket')
        exit(-1)
    for cmd in args.commands:
        getattr(deployer, cmd)()

class Deployer(object):
    def __init__(self, config):
        self.config = config

    def colab(self):
        if self.config.colab['github_repo']:
            bash_fname = os.path.join(os.path.dirname(__file__), 'upload_github.sh')
            run_cmd(['bash', bash_fname, self.config.colab_dir, self.config.colab['github_repo']])

    def sagemaker(self):
        if self.config.sagemaker['github_repo']:
            bash_fname = os.path.join(os.path.dirname(__file__), 'upload_github.sh')
            run_cmd(['bash', bash_fname, self.config.sagemaker_dir, self.config.sagemaker['github_repo']])


class GithubDeployer(Deployer):
    def __init__(self, config):
        super(GithubDeployer, self).__init__(config)
        self.git_dir = os.path.join(self.config.tgt_dir, 'github_deploy')
        shutil.rmtree(self.git_dir, ignore_errors=True)
        mkdir(self.git_dir)

    def html(self):
        run_cmd(['cp -r', os.path.join(self.config.html_dir, '*'), self.git_dir])

    def pdf(self):
        shutil.copy(self.config.pdf_fname, self.git_dir)

    def pkg(self):
        shutil.copy(self.config.pkg_fname, self.git_dir)

    def __del__(self):
        bash_fname = os.path.join(os.path.dirname(__file__), 'upload_github.sh')
        run_cmd(['bash', bash_fname, self.git_dir, self.config.deploy['github_repo']])

class S3Deployer(Deployer):
    def __init__(self, config):
        super(S3Deployer, self).__init__(config)

    def html(self):
        bash_fname = os.path.join(os.path.dirname(__file__), 'upload_doc_s3.sh')
        run_cmd(['bash', bash_fname, self.config.html_dir, self.config.deploy['s3_bucket']])
        self.colab()
        self.sagemaker()

    def pdf(self):
        url = self.config.deploy['s3_bucket']
        if not url.endswith('/'):
            url += '/'
        logging.info('cp %s to %s', self.config.pdf_fname, url)
        run_cmd(['aws s3 cp', self.config.pdf_fname, url, "--acl 'public-read' --quiet"])

    def _deploy_other_files(self, tgt_url):
        other_urls = self.config.deploy['other_file_s3urls'].split()
        for other_url in other_urls:
            logging.info('cp %s to %s', other_url, tgt_url)
            run_cmd(['aws s3 cp', other_url, tgt_url, "--acl 'public-read' --quiet"])

    def pkg(self):
        url = self.config.deploy['s3_bucket']
        if not url.endswith('/'):
            url += '/'
        logging.info('cp %s to %s', self.config.pkg_fname, url)
        run_cmd(['aws s3 cp', self.config.pkg_fname, url, "--acl 'public-read' --quiet"])
        self._deploy_other_files(url)

    def all(self):
        self.html()
        self.pdf()
        self.pkg()
