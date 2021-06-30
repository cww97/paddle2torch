# Rua

记得 `git clone xxxx --recursive`

or `git submodule <init>/<update>`

## 前期准备

```
# download pth
cd models
bash get_pd.sh

# pip install
pip3 install timm, torch
```

## 现在plan

参考 `main.py`: 很暴力

直接读取权重文件然后替换

## 之前plan

### paddle转onnx

```
cd 回来
python3 pd2onnx.py
```

### onnx转torch

```
cp rua.sh onnx2X/
cd onnx2X
bash rua.sh 卡在这里了
```