Dockerfile `amarmet/griffon:latest`

```
runai submit build-remote -i amarmet/griffon:latest --interactive  \
        --service-type=portforward --port 2222:22
```

to have data persistance `--pvc runai-lara-scratch:Container_Mount_Path`


My interactive baby
```
runai submit griffon \
        --image ic-registry.epfl.ch/lara/griffon-build:latest \
        --gpu=1 \
        --cpu=8 \
        --cpu-limit=8 \
        --large-shm \
        --interactive  \
        --environment GIT_USERNAME=SECRET:github,username \
        --environment GIT_PASSWORD=SECRET:github,password \
        --environment WANDB_API_KEY:wandb,key \
        --pvc runai-lara-scratch:/root/scratch \
        --service-type=nodeport --port 2222:22
```

My train baby
```
runai submit griffon \
        --image ic-registry.epfl.ch/lara/griffon-train:latest \
        --gpu=1 \
        --cpu=8 \
        --cpu-limit=8 \
        --large-shm \
        --environment GIT_USERNAME=SECRET:github,username \
        --environment GIT_PASSWORD=SECRET:github,password \
        --environment WANDB_API_KEY=SECRET:wandb,key \
        --pvc runai-lara-scratch:/root/scratch \
        --environment TRAIN_CMD="python src/griffon/models/encoder/count.py --data_root /root/scratch/griffon/data/CounT --config configs/biggest.json --distributed --use_wandb"
```