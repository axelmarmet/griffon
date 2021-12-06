Dockerfile `amarmet/griffon:latest`

```
runai submit build-remote -i amarmet/griffon:latest --interactive  \
        --service-type=portforward --port 2222:22
```

to have data persistance `--pvc runai-lara-scratch:Container_Mount_Path`


My baby
```
runai submit griffon -i amarmet/griffon:latest \
        -g 1 --large-shm \
        --interactive  \
        --pvc runai-lara-scratch:/root/scratch \
        --service-type=portforward --port 2222:22
```