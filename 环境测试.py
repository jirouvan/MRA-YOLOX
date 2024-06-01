import torch

print("torch版本为：",end="")
print(torch.__version__)

print("cuda是否可用？",end="")
print(torch.cuda.is_available())

print("设备有几个GPU？",end="")
print(torch.cuda.device_count())

print("cudnn版本为：",end="")
print(torch.backends.cudnn.version())
print("cuda版本为：",end="")
print(torch.version.cuda,end="")

quit()