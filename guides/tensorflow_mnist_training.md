Training a handwritten digits predicition model using Tensorflow

Login the NAS
```
ssh admin@nas_ip
```

(Option) Confirm the GPU card is mounted
```
GPU=nvidia0 gpu-docker run --rm nvidia/cuda nvidia-smi

Thu Dec 21 10:07:59 2017
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 381.22                 Driver Version: 381.22                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 108...  Off  | 0000:01:00.0     Off |                  N/A |
| 20%   29C    P8     8W / 250W |      2MiB / 11172MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

# Run this to get GPU info every 1 seconds
GPU=nvidia0 gpu-docker run --rm nvidia/cuda nvidia-smi -l 1
```

Start the tensorflow gpu (with python3) docker
```
GPU=nvidia0 gpu-docker run -it --rm -p 16006:6006 -p 18888:8888 tensorflow/tensorflow:1.4.1-gpu-py3 bash
```

If shown error message similar to below, it's mean the port 16006 or 18888 is occupied. Please change to another one, for example, 26006 or 28888.
```
container-station/docker: Error response from daemon: driver failed programming external connectivity on endpoint determined_allen (c76f4c398eb150dadee74ae1398a3339b426b29967398bb813f6fdb00e3ed6bd): Bind for 0.0.0.0:18888 failed: port is already allocated.
```

Start the jupyter
```
jupyter notebook --allow-root
```

You should able to see below message:
```
    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://localhost:8888/?token=08b29a4327e3553a4a68deb5359837b5914e30c5af77f521
```

Open above link, but change the url to nas_ip, 8888 to the port we specified before (18888). For example, http://nas_ip:18888/?token=08b29a4327e3553a4a68deb5359837b5914e30c5af77f521

Download https://github.com/dennischang/quai/raw/master/guides/mnist_deep.ipynb and upload to notebook

Open and run it (Kernel | Run All)

Tensorboard

When running the training, you can see message like Saving graph to: /tmp/model

Using below command to start tensorboard. Change 9aa to your Container Id (only first 3 chars is ok), change /tmp/tmpk9fuday4 to the folder shown in above training steps.
```
tensorboard --logdir=/tmp/model
```

Then open the tensorboard web page. Change the url to nas_ip, change to the port we specified before (16006). For example, http://nas_ip:16006/
