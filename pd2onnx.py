import os
import time
import os.path as osp
import numpy as np

import onnxruntime

import paddle
import paddle2onnx as p2o
from PaddleClas.ppcls.modeling.architectures.resnet_vd import ResNet50_vd


def export_model_to_ONNX():
    model = ResNet50_vd()
    pretrain_path = '/root/paddle/models/ResNet50_vd_10w_pretrained.pdparams'
    state_dict = paddle.load(pretrain_path)
    model.load_dict(state_dict)
    model.eval()

    onnx_path = osp.join('/root/paddle/models', 'resnet50_vd')
    input_spec = paddle.static.InputSpec(shape=[None, 3, 224, 224], dtype='float32', name='input')
    paddle.onnx.export(model, onnx_path, input_spec=[input_spec], opset_version=12, enable_onnx_checker=True)
    return model, onnx_path + '.onnx'


def compare_results(paddle_model, onnx_file):
    x = np.random.random((2, 3, 224, 224)).astype('float32')

    ort_sess = onnxruntime.InferenceSession(onnx_file)
    ort_inputs = {ort_sess.get_inputs()[0].name: x}
    ort_outs = ort_sess.run(None, ort_inputs)
    print("Exported model has been predict by ONNXRuntime!")

    paddle_outs = paddle_model(x)

    np.testing.assert_allclose(ort_outs[0], paddle_outs.numpy(), rtol=1.0, atol=1e-05)
    print("The difference of result between ONNXRuntime and Paddle looks good!")


if __name__ == '__main__':
    paddle_model, onnx_path = export_model_to_ONNX()
    compare_results(paddle_model, onnx_path)
    # import ipdb; ipdb.set_trace()

