python onnx2pytorch.py \
    --onnx_path ../models/resnet50_vd.onnx \
    --simplify_path ../models/resnet50_vd_sim.onnx \
    --pytorch_path ../models/resnet50_vd.pth \
    --input_shape input:1,3,224,224
