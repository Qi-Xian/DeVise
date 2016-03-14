#!/usr/bin/env sh

EXAMPLE=/v2-1/gtlim/lmdb_data/
TOOLS=/home/gtlim/caffe/caffe/build/tools

DATA_ROOT=/home/gtlim/project/word/ebay/
DATA=/home/gtlim/project/txtfile/

if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: DATA_ROOT is not a path to a directory: $DATA_ROOT"
  echo "Set the DATA_ROOT variable in create_word2vec.sh to the path" \
       "where the word2vec is stored."
  exit 1
fi

echo "Creating word2vec lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_word2vec \
    $DATA_ROOT \
    $DATA/ebay_list.txt \
    $EXAMPLE/ebay_word_lmdb
echo "Done."
