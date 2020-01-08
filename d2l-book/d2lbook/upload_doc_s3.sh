#!/bin/bash
#
# Upload a doc folder into a S3 bucket, with text context compressed
#
# Sample Usage:
#
# ./upload_doc_s3.sh build/_build/html/ s3://en.d2l.ai
#
# Requres awscli is installed

set -ex

if [ $# -ne 2 ]; then
    echo "ERROR: needs two arguments. "
    echo "Usage:"
    echo "  $0 doc_dir s3_bucket"
    exit -1
fi

DIR="$( cd $1 && pwd )"
BUCKET=$2
echo "Upload $DIR to $BUCKET"

# use a temp workspace, because we need to modify (compress) some files later.
rm -rf ${DIR}_tmp
cp -r ${DIR} ${DIR}_tmp
DIR=${DIR}_tmp


find $DIR \( -iname '*.css' -o -iname '*.js' \) -exec gzip -9 -n {} \; -exec mv {}.gz {} \;

aws s3 sync --exclude '*.*' --include '*.css' \
     --content-type 'text/css' \
     --content-encoding 'gzip' \
     --acl 'public-read' --quiet \
     $DIR $BUCKET

aws s3 sync --exclude '*.*' --include '*.js' \
     --content-type 'application/javascript' \
     --content-encoding 'gzip' \
     --acl 'public-read' --quiet \
     $DIR $BUCKET

# use a large expire time for fonts
aws s3 sync --exclude '*.*' --include '*.woff' --include '*.woff2' \
     --expires "$(date -d '+24 months' --utc +'%Y-%m-%dT%H:%M:%SZ')" \
     --acl 'public-read' --quiet \
     $DIR $BUCKET

aws s3 sync --delete $DIR $BUCKET --acl 'public-read' --quiet
