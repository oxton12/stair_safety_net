cd ../..
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm \
       -v ./:/workspace/stair_safety_net -it mxnet_gpu
