#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../output" && pwd )"
cd $DIR

FILE=precomputed_ho_rcnn_models.tar.gz
ID=1oGnt-EhuTjjdIcQh1c-GZvz5Ri87Z3xZ

if [ -f $FILE ]; then
  echo "File already exists..."
  exit 0
fi

echo "Downloading precomputed HO-RCNN models (4.0G)..."

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&id=$ID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')&id=$ID" -O $FILE && rm -rf /tmp/cookies.txt

echo "Unzipping..."

tar zxvf $FILE

echo "Done."
