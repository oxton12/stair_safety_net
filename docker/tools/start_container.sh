cd ../..
docker run --gpus all --rm \
    -v ./:/workspace/Ultralight-SimplePose \
    -it mxnet_gpu
