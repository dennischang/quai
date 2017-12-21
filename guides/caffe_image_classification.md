# Image Classification using Caffe and QuAI

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

Start the caffe:gpu docker and run the first example
```
GPU=nvidia0 gpu-docker run -it bvlc/caffe:gpu bash

cd /opt/caffe

ls -l models/bvlc_googlenet/

./scripts/download_model_binary.py models/bvlc_googlenet

./data/ilsvrc12/get_ilsvrc_aux.sh

./build/examples/cpp_classification/classification.bin models/bvlc_googlenet/deploy.prototxt models/bvlc_googlenet/bvlc_googlenet.caffemodel data/ilsvrc12/imagenet_mean.binaryproto data/ilsvrc12/synset_words.txt examples/images/cat.jpg
```

Write the first python program to predict
```
# install vim
cd
touch /var/cache/apt/archives/lock
rmdir /var/cache/apt/archives/partial
mkdir -p /var/cache/apt/archives/partial
apt update
apt install vim -y
```

Add a classify.py python program
```
vi classify.py
```

Code:
```
import os
import caffe
import numpy as np
import datetime
import time
import sys
model = '/opt/caffe/models/bvlc_googlenet/deploy.prototxt'
weights = '/opt/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel'
if len(sys.argv)<2:
  print 'example: python classify.py image.jpg'
  sys.exit()
caffe.set_mode_gpu()
caffe.set_device(0)
caffeNet = caffe.Net(model, weights, caffe.TEST)
img = caffe.io.load_image(sys.argv[1])
img = caffe.io.resize_image(img, (224, 224, 3))
img = img[:,:,::-1]*255.0
img = img.transpose((2,0,1))
img = img[None,:]
tic = time.time()
res = caffeNet.forward_all(data = img)
toc = time.time()
elapsed = toc - tic
output_prob = caffeNet.blobs['prob'].data[0]
top_inds = output_prob.argsort()[::-1][:5]
labels_file = '/opt/caffe/data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(labels_file, str, delimiter='\t')
result = np.vstack((labels[top_inds], output_prob[top_inds])).T
print('Elapsed: ' + str(elapsed*1000) + 'ms')
print(result.tolist())
```

Run it:
```
python classify.py /opt/caffe/examples/images/cat.jpg
```

Use CPU instead, mark below 2 lines, and run again
```
#caffe.set_mode_gpu()
#caffe.set_device(0)
```



