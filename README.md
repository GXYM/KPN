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
