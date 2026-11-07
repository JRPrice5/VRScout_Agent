import os


def count_jpgs(root_dir):
    jpg_count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                jpg_count += 1
    return jpg_count

if __name__ == "__main__":
    directory = r"D:\CLOVR_Output\PW_DATA_3\Data"  # change this to your directory
    total_jpgs = count_jpgs(directory)
    print(f"Total JPG files under '{directory}': {total_jpgs}")
