from pathlib import Path
import glob
import argparse
from box import Box

parser = argparse.ArgumentParser(description="sum the integers at the command line")
parser.add_argument("root", type=str, help="root of the file name")
try:
    args = parser.parse_args()
except:
    args = Box({"root": "datasets/aihub/1.Training/frames/시나리오34/카메라*"})

if __name__ == "__main__":
    file_list = sorted(glob.glob(f"{args.root}/*.jpg"))
    for file_path in file_list:
        # "sunny"를 "cloudy"로 변경
        new_file_path = file_path.replace("sunny", "cloudy")
        Path(file_path).rename(new_file_path)
    print("Done!")
