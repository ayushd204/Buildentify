import os
import random
import shutil

# move images to train and test folders

basefolder = "E:/home/works/Desktop/projects/vision/dataset"

copy=True
percentage=0.02
folder_name="/test"
proceed=True

images_folder = basefolder+"/MergeDataset"
test_folder = basefolder+folder_name

subfolders = [f.path for f in os.scandir(images_folder) if f.is_dir()]

def move_files(src_folder, dest_folder, files):
    for file in files:
        os.rename(os.path.join(src_folder, file), os.path.join(dest_folder, file))

def copy_files(src_folder, dest_folder, files):
    for file in files:
        shutil.copy(os.path.join(src_folder, file), os.path.join(dest_folder, file))

for folder in subfolders:
    folder_name = os.path.basename(folder)
    images=os.listdir(folder)

    test_files = random.sample(images, int(len(images)*percentage))

    if copy:
        print(f"copying {len(test_files)} files from {folder_name} to test folder")
    else:
        print(f"moving {len(test_files)} files from {folder_name} to test folder")
    if not proceed:
        input("press enter to continue")
    
    # move the test files to the test folder
    test_folder_for_building = os.path.join(test_folder, folder_name)
    os.makedirs(test_folder_for_building, exist_ok=True)

    for file in test_files:
        # move files into test folder
        if copy_files:
            shutil.copy(os.path.join(folder, file), os.path.join(test_folder_for_building, file))
        else:
            os.rename(os.path.join(folder, file), os.path.join(test_folder_for_building, file))
