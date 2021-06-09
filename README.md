# Rua

记得 `git clone xxxx --recursive`

or `git submodule <init>/<update>`

## 下载模型文件

```
cd models
bash get_pd.sh
```

## paddle转onnx

```
cd 回来
python3 pd2onnx.py
```

## onnx转torch

```
cp rua.sh onnx2X/
cd onnx2X
bash rua.sh 卡在这里了
```