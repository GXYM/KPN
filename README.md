# KPN
The code of *Kernel Proposal Network for Arbitrary Shape Text Detection*.  
The paper has been  accepted by TNNLS 2022, which is available at: https://arxiv.org/abs/2203.06410

# Prerequisites
PyTorch >= 1.2.0
torchvision

cycler
easydict
matplotlib
numpy
opencv-python
Pillow
Polygon3
scipy
Shapely
tensorboardX
tqdm

# Image
Put your texting images in the floders of 
```
"../../data_model/data/total-text-mat/Images/Test/"
"../../data_model/data/ctw1500/test/text_image/"
"../../data_model/data/Icdar2015/Test/" 
```

You can change the relative path "../../data_model/" in util/config.py: config.data_model_path = "../../data_model/"


# Run
You can get the model weight at [model](https://drive.google.com/file/d/1WvJUTggqYXBkKtu3vSvIJQ_A7b7ZYER9/view?usp=sharing). And unzip them in current path.

You can run the code by:
```
sh eval_totaltext.sh [gpu ID]
sh eval_ctw1500.sh [gpu ID]
sh eval_IC15.sh [gpu ID]
```
if you want to view the separate text instances, you can add "--eval_vis True" in the above ".sh", it will show the separate text instances with the OpenCV "cv2.imshow".

The results will be stored in the floder "vis".

# Citing the related works

```
@article{zhang2022kernel,
  title={Kernel proposal network for arbitrary shape text detection},
  author={Zhang, Shi-Xue and Zhu, Xiaobin and Hou, Jie-Bo and Yang, Chun and Yin, Xu-Cheng},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022},
  publisher={IEEE}
}

@inproceedings{DBLP:conf/cvpr/ZhangZHLYWY20,
  author       = {Shi{-}Xue Zhang and
                  Xiaobin Zhu and
                  Jie{-}Bo Hou and
                  Chang Liu and
                  Chun Yang and
                  Hongfa Wang and
                  Xu{-}Cheng Yin},
  title        = {Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection},
  booktitle    = {2020 {IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
                  {CVPR} 2020, Seattle, WA, USA, June 13-19, 2020},
  pages        = {9696--9705},
  publisher    = {Computer Vision Foundation / {IEEE}},
  year         = {2020},
  doi          = {10.1109/CVPR42600.2020.00972},
}

@inproceedings{DBLP:conf/iccv/Zhang0YWY21,
  author    = {Shi{-}Xue Zhang and
               Xiaobin Zhu and
               Chun Yang and
               Hongfa Wang and
               Xu{-}Cheng Yin},
  title     = {Adaptive Boundary Proposal Network for Arbitrary Shape Text Detection},
  booktitle = {2021 {IEEE/CVF} International Conference on Computer Vision, {ICCV} 2021, Montreal, QC, Canada, October 10-17, 2021},
  pages     = {1285--1294},
  publisher = {{IEEE}},
  year      = {2021},
}

@article{zhang2023arbitrary,
  title={Arbitrary shape text detection via boundary transformer},
  author={Zhang, Shi-Xue and Yang, Chun and Zhu, Xiaobin and Yin, Xu-Cheng},
  journal={IEEE Transactions on Multimedia},
  year={2023},
  publisher={IEEE}
}

@article{DBLP:journals/pami/ZhangZCHY23,
  author       = {Shi{-}Xue Zhang and
                  Xiaobin Zhu and
                  Lei Chen and
                  Jie{-}Bo Hou and
                  Xu{-}Cheng Yin},
  title        = {Arbitrary Shape Text Detection via Segmentation With Probability Maps},
  journal      = {{IEEE} Trans. Pattern Anal. Mach. Intell.},
  volume       = {45},
  number       = {3},
  pages        = {2736--2750},
  year         = {2023},
  url          = {https://doi.org/10.1109/TPAMI.2022.3176122},
  doi          = {10.1109/TPAMI.2022.3176122},
}

```
