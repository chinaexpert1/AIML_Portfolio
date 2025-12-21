import torch
print("torch:", torch.__version__, "cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
# minimal CUDA ops
x = torch.randn(2,2, device="cuda")
y = torch.arange(10, device="cuda")
print((x@x).shape, y[:3])
