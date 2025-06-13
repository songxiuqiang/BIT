# (BIT) Prior-free 3D Object Tracking

[[Website]](https://github.com/songxiuqiang/BIT) [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Song_Prior-free_3D_Object_Tracking_CVPR_2025_paper.pdf) [[Video]](https://www.youtube.com/watch?v=C3CoXe6Kkt0&t=1s)

This is the official implementation of paper **Prior-free 3D Object Tracking (CVPR 2025 Highlight)**


<p align="center">
  <img src="/assets/img/top-2.png" width="35%">
  <img src="/assets/img/top-800.gif" width="64%">  
</p>

---

**BIT**(**B**idirectional **I**terative **T**racking) is a truly **prior-free 3D object tracking** method that does not rely on any pre-defined 3D models or training priors during the tracking process. Our approach consists of a geometry generation module and a pose optimization module. The core idea is to enable these two modules to automatically and iteratively enhance each other, thereby progressively building all the necessary information required for the tracking task. For this reason, we refer to our method as Bidirectional Iterative Tracking (BIT).

## Future work

- [ ] Improve the communication mechanism between Python and C++
- [ ] A more industrialized and user-friendly version may be provided in the future
- [ ] Docker deployment

## Installation

### Environment

The code was originally developmented on 

* Ubuntu 20.04 with Python 3.9 and CUDA 11.6, using an NVIDIA RTX 3080 GPU.

[@winka9587](https://github.com/winka9587) helped test it on 
* Ubuntu 22.04 with Python 3.10 and CUDA 11.8, using an NVIDIA RTX 4090 GPU. 
* Ubuntu 20.04 (laptop) with Python 3.10 and CUDA 11.8, using an NVIDIA RTX 4090 GPU (laptop). 

As Python 3.10 and CUDA 11.8 are more convenient and newer, we **recommend** running the code in this environment.

1. Clone the repo

~~~bash
git clone https://github.com/songxiuqiang/BIT
~~~

#### Python
2. Create the environment, activate it and install dependencies:

~~~bash
conda create -n BIT_Track python=3.10
conda activate BIT_Track
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python setup.py install   # install soft_render
~~~

#### Checkpoints

~~~bash
mkdir checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
~~~

#### C++

We tested our codes with the following dependencies:
~~~ bash
OpenCV=4.0/4.5
Eigen=3.3.9/3.4.0
Qt=5.12.7
~~~

We use the baseline [tracker](https://github.com/huanghone/SLCT) (referred to as SLOT in our paper and refer to Summer in the codes) to track the pose. You can compile it by the script below. If you encounter missing libraries, please install them using `apt` or build them from source as referenced in the GitHub repository. 

~~~bash
# in the root directory of the repo
sh setup_cpp.sh
~~~



### DATASET

|dataset|size|website|code|download link|
|--|--|--|--|--|
|**RBOT**|(~40GB)| [website](https://www.mi.hs-rm.de/~schwan/research/RBOT/)|[henningtjaden/RBOT](https://github.com/henningtjaden/RBOT)|[download link](https://www.mi.hs-rm.de/~schwan/research/RBOT/files/RBOT_dataset.zip)|
|**MOPED**|(~6GB)|[website](https://latentfusion.github.io/) |[NVlabs/latentfusion](https://github.com/NVlabs/latentfusion)| [download link](https://huggingface.co/datasets/winka9587/MOPED/tree/main)|

For details, see [README_DATASET.md](README_DATASET.md).

## Applications

**1. Generate Configs**

This step will generate configuration files required for testing on the [MOPED](https://github.com/NVlabs/latentfusion) and [RBOT](https://github.com/henningtjaden/RBOT) datasets, the generated config files will be stored in the `config/moped/`, `config/rbot/` and etc.

~~~bash
# Edit the data path in config/dataset.yml.
moped_dir: "/path/to/dataset/moped"
rbot_dir: "/path/to/dataset/RBOT_dataset"
~~~
Run the script to generate configs:
~~~bash
sh gen_configs.sh
~~~

> You can also seperatly generate config for single dataset by uncomment code in ``gen_configs.sh``. If you want change some parameters, such as the number of reference frames, you can modify the ``config/config_generator.py`` file.

**2. Generate Reference Frames**

Run the script to generate reference frames: 

~~~bash
# sh gen_refers.sh $dataset_name $config_file 

# for example, MOPED dataset (both needed)
sh gen_refers.sh "moped" "black_drill/reference/00.yml"
sh gen_refers.sh "moped" "black_drill/evaluation/00.yml"

# RBOT dataset
sh gen_refers.sh "rbot" "a_regular_ape.yml"
~~~

**Note**: The number of reference frames can be changed by setting the variable `reference_imgs_nums` in the file `config/config_generator.py`.

Reference frames are optional in our method and not strictly required. However, for unified framework implementation, the first frame is treated as a pseudo reference frame to maintain consistency with the actual reference frames. As a result, this step is necessary in the code, and the number of reference frames should be set to one more than the actual number.


**3. Tracking**

**Randomness**  We find that the differential rendering results have a tiny amount of randomness, which may be caused by the precision of float values in its CUDA code. However, you should get results close to those reported in our paper.


~~~bash
# sh run_example.sh $1 $2
sh run_example.sh "moped" "black_drill/evaluation/00.yml" 
~~~

## Citation

~~~
@inproceedings{song2025bit,
  title={Prior-free 3D Object Tracking},
  author={Song, Xiuqiang and Jin, Li and Zhang, Zhengxian and Li, Jiachen and Zhong, Fan and Zhang, Guofeng and Qin, Xueying},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025},
  publisher={IEEE}
}
~~~

## Acknowledgements

We thank [@winka9587](https://github.com/winka9587) for his help in testing the code on different platforms and providing valuable feedback.

## Licenses

BIT is licensed under the GNU General Public License Version 3 (GPLv3), see [http://www.gnu.org/licenses/gpl.html](http://www.gnu.org/licenses/gpl.html).