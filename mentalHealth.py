import kagglehub
import shutil
import os

path = kagglehub.dataset_download("bhavikjikadara/mental-health-dataset")
print("Original path (cache):", path)

target_dir = os.getcwd()
for file in os.listdir(path):
    src = os.path.join(path, file)
    dst = os.path.join(target_dir, file)
    shutil.copy(src, dst)

print("Files copied to:", target_dir)

# pyhton mentalHealth.py # to run the code and get dataset