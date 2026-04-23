import os
import shutil

# --- CONFIGURATION ---
# Define the path to mother directory containing files in need of renaming
TARGET_ROOT = "/NAS/Jon/pGRAMS_FlightBoardZ/Run7" 

# Map for possible desired tags to use in file renaming
TAG_MAP = {
    "light-pedestal": "light-pedestal",
    "light-sig":      "light-sig",
    "combo":          "combo",
    "charge-pedestal": "charge-pedestal"
}

def fix_filenames():
    print(f"Scanning {TARGET_ROOT} for files needing tags...")
    count = 0
    
    for root, dirs, files in os.walk(TARGET_ROOT):
        folder_name = os.path.basename(root)
        
        # First check if the directory has one of the tags we desire
        found_tag = None
        for key, tag in TAG_MAP.items():
            if key in folder_name:
                found_tag = tag
                break
        
        if not found_tag:
            continue
            
        # If folder has tag, check the FILES inside
        for filename in files:
            if not filename.endswith('.bin'):
                continue
                
            # If the file already has the tag, skip it
            if found_tag in filename:
                continue
            
            # Construct the new name
            #  Find "dig2" or "dig1" and insert the tag before it.
          
            if "_dig" in filename:
                new_filename = filename.replace("_dig", f"_{found_tag}_dig")
            else:
                # Fallback: Just append it to the end before extension if 'dig' pattern misses
                base, ext = os.path.splitext(filename)
                new_filename = f"{base}_{found_tag}{ext}"
            
            # 4. Rename
            old_path = os.path.join(root, filename)
            new_path = os.path.join(root, new_filename)
            
            print(f"[RENAME] {filename} \n      -> {new_filename}")
            os.rename(old_path, new_path)
            count += 1

    print(f"---------------------")
    print(f"Renaming Complete. Modified {count} files.")

if __name__ == "__main__":
    confirm = input(f"About to rename files in {TARGET_ROOT}. Type 'yes' to proceed: ")
    if confirm.lower() == 'yes':
        fix_filenames()
    else:
        print("Aborted.")