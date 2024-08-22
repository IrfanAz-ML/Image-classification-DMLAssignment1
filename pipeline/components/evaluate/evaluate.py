
evaluate_script = """\
import torch
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

def evaluate_model(data_dir, model_dir):
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    val_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = torch.load(model_dir + '/model.pth')
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Validation Accuracy: {accuracy}')

if __name__ == "__main__":
    evaluate_model('/mnt/dataset/validation', '/mnt/model')
"""

# Save the script to the appropriate directory
with open("pipeline/components/evaluate/evaluate.py", "w") as file:
    file.write(evaluate_script)

print("Evaluation script created successfully!")
