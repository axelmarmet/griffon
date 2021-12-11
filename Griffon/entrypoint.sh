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
    pip install -e .
)

cd griffon/Griffon

if [[ -z "${TRAIN_CMD}" ]]; then
    eval $TRAIN_CMD
fi