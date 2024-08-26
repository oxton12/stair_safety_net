cd ../..
docker run --gpus all --rm \
    -v ./:/workspace/stair_safety_net \
    -it mxnet_gpu
