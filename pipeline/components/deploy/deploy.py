
deploy_script = """\
from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)

model = models.resnet18()
model.load_state_dict(torch.load('/mnt/model/model.pth'))
model.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        outputs = model(tensor)
        _, predicted = outputs.max(1)
        return jsonify({'prediction': predicted.item()})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
"""

# Save the script to the appropriate directory
with open("pipeline/components/deploy/deploy.py", "w") as file:
    file.write(deploy_script)

print("Deployment script created successfully!")
