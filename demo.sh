set -e

mkdir -p scratch && cd scratch

python ../project2_util.py blend ../example1/apple.jpg ../example1/orange.jpg ../example1/split_mask.png example1a.jpg

python ../project2_util.py show ../example2/henson.json
python ../project2_util.py crop ../example2/henson.json 2.2 3.0 1000 0 50 henson_crop.jpg henson_crop.json
python ../project2_util.py show ../example2/johnson.json
python ../project2_util.py warp ../example2/johnson.json henson_crop.json johnson_crop.jpg
python ../project2_util.py overlay henson_crop.jpg johnson_crop.jpg 

python ../project2_util.py show ../example3/cushing.json
python ../project2_util.py ellipse ../example3/cushing.json 1.4 2.2 cushing_mask.png
python ../project2_util.py show ../example3/prince.json
python ../project2_util.py warp -flip ../example3/prince.json ../example3/cushing.json prince_warp.jpg
python ../project2_util.py blend ../example3/cushing.jpg prince_warp.jpg cushing_mask.png example3a.jpg

python ../project2_util.py show ../example4/paul.json
python ../project2_util.py ellipse ../example4/paul.json 1.7 2.3 paul_mask.png 
python ../project2_util.py show ../example4/elizabeth.json
python ../project2_util.py ellipse ../example4/elizabeth.json 1.7 2.3 elizabeth_mask.png 
python ../project2_util.py warp -flip ../example4/paul.json ../example4/elizabeth.json elizabeth_replacement.jpg 
python ../project2_util.py warp -flip ../example4/elizabeth.json ../example4/paul.json paul_replacement.jpg 
python ../project2_util.py blend paul_replacement.jpg elizabeth_replacement.jpg ../example4/split_mask.png wandavision_faces_swapped.jpg
python ../project2_util.py blend paul_mask.png elizabeth_mask.png ../example4/split_mask.png wandavision_masks.png
python ../project2_util.py blend ../example4/wandavision.jpg wandavision_faces_swapped.jpg wandavision_masks.png example4a.jpg
