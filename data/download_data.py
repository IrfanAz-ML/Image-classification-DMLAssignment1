
download_data_script = """\
import os
from torchvision import datasets, transforms

def download_imagenet_data(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    datasets.ImageNet(root=data_dir, split='train', download=True, transform=transform)

if __name__ == "__main__":
    download_imagenet_data('/mnt/dataset')
"""

# Save the script to the appropriate directory
with open("data/download_data.py", "w") as file:
    file.write(download_data_script)

print("Data download script created successfully!")
