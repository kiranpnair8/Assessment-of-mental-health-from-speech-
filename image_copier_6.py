import os
import shutil

source_folder ="C:\\project main\\Speech_Dirization\\train"
target_folder = "C:\\project main\\Speech_Dirization\\Train_images"

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

png_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.endswith('.png')]


for png_file in png_files:
    shutil.copy(png_file, target_folder)