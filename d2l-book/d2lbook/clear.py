import logging
import shutil
from d2lbook.config import Config

__all__  = ['clear']

def clear():
    config = Config()
    build_dir = config.tgt_dir
    logging.info('Delete %s', build_dir)
    shutil.rmtree(build_dir, ignore_errors=True)
