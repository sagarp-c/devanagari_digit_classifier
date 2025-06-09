import os

# Path to the numerals directory
base_dir = 'numerals'

# Loop through folders 0 to 9
for digit in map(str, range(10)):
    folder_path = os.path.join(base_dir, digit)
    
    if not os.path.isdir(folder_path):
        print(f"Folder {folder_path} does not exist, skipping.")
        continue
    
    # List all files in folder (filtering only image files - common formats)
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # Sort files to maintain order (optional)
    files.sort()
    
    # Rename files
    for i, filename in enumerate(files, start=1):
        ext = os.path.splitext(filename)[1]  # get file extension
        new_name = f"digit_{digit}_{i}{ext}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")

print("Renaming complete!")
