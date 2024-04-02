import re


def find(pattern, string):
    return re.findall(pattern, string)[0]


# AIHub dataset
# 시나리오01/카메라11.avi
#         ^~       ^~
scenario_id_pattern = r"(?<=시나리오)\d{2}"
camera_id_pattern = r"(?<=카메라)\d{2}"
frame_id_pattern = r"_(\d{4})."
frame_id_pattern_f = r"f(\d{4})."
