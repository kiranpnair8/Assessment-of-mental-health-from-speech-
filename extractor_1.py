import os
import zipfile

zip_folder = 'C:\project main\Speech_Dirization\data' 
# Loop through all zip files in the folder
for zip_file in os.listdir(zip_folder):
    print(zip_file)
    if zip_file.endswith('.zip'):  # Check if the file is a zip file
        # Create a folder with the same name as the zip file
        folder_name = os.path.splitext(zip_file)[0]
        folder_path = os.path.join(zip_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Extract the contents of the zip file to the folder
        zip_path = os.path.join(zip_folder, zip_file)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(folder_path)