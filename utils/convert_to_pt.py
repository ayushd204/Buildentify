import os
import torch
from torchvision import transforms
from PIL import Image
import shutil
import random

# Set paths
input_folder = "E:/home/works/Desktop/projects/vision/dataset/MergeDataset/"
output_folder = "E:/home/works/Desktop/projects/vision/dataset/mergedataset_tensors/"
os.makedirs(output_folder, exist_ok=True)

transform = transforms.ToTensor()

# get a list of folders in the input folder
list_of_folders = os.listdir(input_folder)

# paths=[ os.path.join(output_folder,folder) for folder in list_of_folders]

made_folder = []
# make these folders in the output folder
for folder in list_of_folders:
    made_folder.append(os.path.join(output_folder, folder))
    os.makedirs(made_folder[-1], exist_ok=True)

# Process each image
def convert_and_copy(source,dest):
    if source.endswith(".png"):
        # Load, transform, and save
        image = Image.open(source).convert("RGB")
        tensor = transform(image)
        print(f"Saving {dest}")
        torch.save(tensor, dest)


move_dict={} # source path: dest path

souce_folder="E:/home/works/Desktop/projects/vision/dataset/MergeDataset/Carpenter Hall/"
dest_folder="E:/home/works/Desktop/projects/vision/dataset/mergedataset_pt/Carpenter Hall/"

source_files=os.listdir(souce_folder)
source_file= random.shuffle(source_files)
files_already_converted=os.listdir(dest_folder)

c=0
for source_file in source_files:
    if source_file in files_already_converted:
        continue
    else:
        c+=1
        source_file_path=os.path.join(souce_folder, source_file)
        dest_file_path=os.path.join(dest_folder, source_file)
        move_dict[source_file_path]=dest_file_path
        # print(f"moving {source_file_path} to {dest_file_path}")
        # input("Press Enter to continue...")
        convert_and_copy(source_file_path, dest_file_path)
        if c>=100:
            break
print(c)



# get a list of files in each folder
# for folder in list_of_folders:
#     orginal_folder_path = os.path.join(input_folder, folder)
#     list_of_files = os.listdir(orginal_folder_path)

#     for file in list_of_files:
#         source_file_path=os.path.join(output_folder, file)
#         source_file_path=source_file_path.replace('.png','.pt')

#         dest_file_path=os.path.join(output_folder,folder,file)

#         move_dict[source_file_path]=dest_file_path

# print(len(move_dict))

# count=0
# for source, dest in move_dict.items():
#     shutil.move(source, dest)
#     if count%1000==0:
#         print(f"Moved {count}")





