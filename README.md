# EVLoc
This repository contains the source code for our paper:

EVLoc: Event-based Visual Localization in LiDAR Maps via Event-Depth Registration<br/>
Kuangyi Chen, Jun Zhang, Friedrich Fraundorfer<br/>

## Requirements
The code has been trained and tested with PyTorch 2.2.2 and Cuda 12.2.
```Shell
conda create -n E2D python=3.10 -y
conda activate E2D
pip install -r requirements.txt
cd core/correlation_package
python setup.py install
cd ..
cd visibility_package
python setup.py install
cd ../..
```

## Required Data
To evaluate/train EVLoc, you could download the M3ED dataset.
* [M3ED](https://m3ed.io/data_overview/)

We trained and tested EVLoc on the M3ED Falcon Indoor sequences flight_1, flight_2, and flight_3.

To obtain the required data for direct use, we sample a pre-defined range of point cloud from the whole LiDAR maps for every provided pose.
```Shell
python tools/map2pc.py --dataset [PATH_TO_DATA] --sequence [SPECIFIED_SEQUENCE] --save_dir [PATH_TO_SAVEDIR]
```
As for event data, we also need to generate event frames accordingly.
```Shell
python tools/event2frame.py --dataset [PATH_TO_DATA] --sequence [SPECIFIED_SEQUENCE] --save_dir [PATH_TO_SAVEDIR] --camera [WHICH_CAMERE] --method [EVENT_REPRESENTATION]
```





## Demos
Pretrained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1ASoSb4XsNDopaBkD503m9Vo_UHYpGTsa?usp=sharing)

You can demo a trained model on a sequence of frames
```Shell
python main_event_baseline.py --ev_input ours_denoise_pre_100000 --load_checkpoints checkpoints/baseline.pth -e
```

