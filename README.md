# lighttrack

Clone from https://github.com/Guanghan/lighttrack.

## 環境構築

GPU必須。

### Prerequisites


```bash
$ conda create -n lighttrack python=3.7
$ conda activate lighttrack
$ conda install tensorflow-gpu=1.14 pillow=6.2.1
$ conda install -c conda-forge tqdm setproctitle matplotlib
$ conda install -c anaconda cython
```

condaで入らないものはpipでいれる。pipのバージョンが3.7であることに注意して、
```
$ pip install opencv-python==4.2.0 pyyaml==5.3
$ pip install torch==1.4.0 torchvision==0.5.0
```

### Getting Started

元のREADMEに従う。
