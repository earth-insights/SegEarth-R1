# Installation⚙️

Set up conda envirnment:

```bash
conda create --name=segearthr1 python=3.10
conda activate segearthr1

git clone https://github.com/earth-insights/SegEarth-R1.git
cd segearthr1

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

pip install -r requirements.txt
```