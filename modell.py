import zipfile

zip_path = "vosk-model-small-en-us-0.15.zip"
dest_folder = "vosk-model-small-en-us-0.15"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(dest_folder)

print("Model unzipped in folder:", dest_folder)