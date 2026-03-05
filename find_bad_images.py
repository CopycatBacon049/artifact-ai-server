import os
from PIL import Image

root = "artifacts"

for folder, _, files in os.walk(root):
    for file in files:
        path = os.path.join(folder, file)
        try:
            img = Image.open(path)
            img.verify()
        except Exception as e:
            print("BAD FILE:", path, "| ERROR:", e)
