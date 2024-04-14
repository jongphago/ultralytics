rm -r datasets/aihub/4.Small
mkdir datasets/aihub/4.Small
mkdir datasets/aihub/4.Small/labels
mkdir datasets/aihub/4.Small/images
cp -r /home/jongphago/project/ultralytics/datasets/aihub/2.Validation/labels/*_f0000.txt datasets/aihub/4.Small/labels/
cp -r datasets/aihub/2.Validation/frames/시나리오*/카메라*/*_f0000.jpg datasets/aihub/4.Small/images/
