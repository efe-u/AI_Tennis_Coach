import os

FOLDERS = {
    "images": "_annotated"
}

for folder in FOLDERS:
    for i in range(len(os.listdir(folder))):
        for img in os.listdir(folder):
            os.remove(folder+"/"+img)
