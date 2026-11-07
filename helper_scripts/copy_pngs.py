import os
import shutil


def collect_pngs(src_root, dst_root):
    """
    Copies all PNG images from subfolders in src_root to dst_root,
    preserving the subfolder structure.
    
    :param src_root: The source folder path
    :param dst_root: The destination folder path
    """
    for root, dirs, files in os.walk(src_root):
        # Get the relative path from the source root
        rel_path = os.path.relpath(root, src_root)
        
        # Destination path with preserved subfolder structure
        dst_path = os.path.join(dst_root, rel_path)
        
        # Make sure destination folder exists
        os.makedirs(dst_path, exist_ok=True)
        
        for file in files:
            if file.lower().endswith(".png"):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_path, file)
                
                # Copy PNG file
                shutil.copy2(src_file, dst_file)
                print(f"Copied: {src_file} -> {dst_file}")

if __name__ == "__main__":
    # Example usage
    src_folder = r"D:\PythonProjects\Project-VR\ckpts"
    dst_folder = r"D:\PNGs"
    collect_pngs(src_folder, dst_folder)
