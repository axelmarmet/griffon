#!/bin/bash

echo "https://${GIT_USERNAME}:${GIT_PASSWORD}@github.com" > .credentials.txt

git config --global credential.helper 'store --file .credentials.txt'

# clone and install CoqGym
git clone https://github.com/axelmarmet/CoqGym.git
(
    cd CoqGym
    pip install -e .
)

# clone and install griffon
git clone https://github.com/axelmarmet/griffon.git
(
    cd griffon/Griffon
    git checkout lightning
    pip install -e .
)

cd griffon/Griffon

eval $TRAIN_CMD
