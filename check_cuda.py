import torch

num_gpus = torch.cuda.device_count()

names = []
for i_gpu in range(num_gpus):
    names.append(torch.cuda.get_device_name(i_gpu))

print("Checking PyTorch CUDA devices...")
print("torch.cuda.is_available():", torch.cuda.is_available())
print(f"Found {num_gpus} GPUs")
print("Device names: ", names)
print("PyTorch CUDA check complete")
