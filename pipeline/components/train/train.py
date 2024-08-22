
train_script = """\
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms

def train_model(data_dir, model_dir):
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1000)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

    torch.save(model.state_dict(), model_dir + '/model.pth')

if __name__ == "__main__":
    train_model('/mnt/dataset/preprocessed', '/mnt/model')
"""

# Save the script to the appropriate directory
with open("pipeline/components/train/train.py", "w") as file:
    file.write(train_script)

print("Training script created successfully!")
