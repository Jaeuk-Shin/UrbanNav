# Data Preparation

## CityWalk Video Dataset
In our data processing pipeline, we first divide the long videos into short clips of two minutes to reduce IO bottleneck during training. We then use [DPVO](https://github.com/princeton-vl/DPVO) to get the odometry in each clip. Therefore we cannot provide the trajectory for each video. We only provide the playlists of the videos we used and our processing pipeline.

### Download
Please download the videos in `citywalk_playlists.txt` and/or `citydrive_playlists.txt`(optional). And put them in a single directory. We recommend using [youtube-dl](https://github.com/ytdl-org/ytdl-nightly) to download.

### Preprocessing
#### 1. Split the videos
Split the videos into clips of 2 minutes. If you have a slurm system, we recommend using `utils/video_split/run_split.sh` to submit array jobs for parallel processing:
```
sbatch utils/video_split/run_split.sh
```
Otherwise, you can use `utils/video_split/split.py` to split the videos sequentially.

#### 2. Get odometry
First, donwload the DPVO submodule by
```
git submodule update --init --recursive
```
Then, install the environment for DPVO by following the instructions in `DPVO/README.md` (no need to install Pangolin Vidwer or Classical Backend).

Now you can submit array jobs for parallel processing:
```
sbatch third_party/DPVO/run_dpvo_slurm.sh
```

## Teleportation Dataset
We also provide the teleportation dataset for finetuning and testing. The dataset can be donwloaded at our Hugging Face dataset page: [ai4ce/CityWalker](https://huggingface.co/datasets/ai4ce/CityWalker).
