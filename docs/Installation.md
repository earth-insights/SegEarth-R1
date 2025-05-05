# Installation⚙️

## Requirement:

* Linux with Python ≥ 3.10.
* PyTorch ≥ 2.0 and torchvision that matches the PyTorch installation.

## Set up conda envirnment:
```bash
conda create -n segearthr1 python=3.10
conda activate segearthr1

git clone https://github.com/earth-insights/SegEarth-R1.git
cd segearthr1

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```
## Install detectron2:
Follow [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

## CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

```
cd segearth_r1/model/mask_decoder/Mask2Former_Simplify/modeling/pixel_decoder/ops
sh make.sh
```
