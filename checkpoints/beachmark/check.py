import onnx

onnx_model = onnx.load("models.onnx")

try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print("onnx model is incorrect")
else:
    print("onnx model is correct")