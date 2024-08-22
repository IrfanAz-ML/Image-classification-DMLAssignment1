
preprocess_script = """\
import os
import cv2

def preprocess_image(image_path, output_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    cv2.imwrite(output_path, img)

def preprocess_dataset(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        preprocess_image(input_path, output_path)

if __name__ == "__main__":
    preprocess_dataset('/mnt/dataset/train', '/mnt/dataset/preprocessed')
"""

# Save the script to the appropriate directory
with open("pipeline/components/preprocess/preprocess.py", "w") as file:
    file.write(preprocess_script)

print("Preprocessing script created successfully!")
