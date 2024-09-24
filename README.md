## Multi-agent Exploration with Similarity Score Map and Topological Memory
Eun Sun Lee and Young Min Kim<br />
3D Vision Lab, Seoul National Unversity

Project Website: 

![example](./docs/example.gif)

### Overview:
We propose a multi-agent exploration strategy that coordinates the actions of multiple robots within a shared environment <br />
to optimize overall exploration coverage. 
Our approach combines the classical frontier-based method with a novel similarity score map derived from topological graph memory.


## Installing Dependencies

Installing habitat-sim:
```
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim; git checkout 9575dcd45fe6f55d2a44043833af08972a7895a9;
pip install -r requirements.txt;
python setup.py install --headless

```

Installing habitat-api:
```
git clone https://github.com/facebookresearch/habitat-api.git
cd habitat-api; git checkout b5f2b00a25627ecb52b43b13ea96b05998d9a121;
pip install -e .
```

Installing pytorch: 
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.0 -c pytorch -c nvidia
```

## Setup
Clone the repository and install other requirements:
```
git clone --recurse-submodules https://github.com/eunsunlee/SimilarityScoreMap
cd SimilarityScoreMap;
pip install -r requirements.txt
```

Dataset can be downloaded from the following link: https://github.com/facebookresearch/habitat-api#data
