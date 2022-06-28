Credits and copyright information for all example images can be found
in `credits.txt` files in each `example` directory.

# E27 Project 2

Before you begin programming, I strongly encourage you to work through
the examples below by running the commands listed.

All of the examples below use `project2_util.py` file, which can act
both as a script you can run from the command line, as well as a module
you can `import` into your own programs. 

To get command-line help, just run the file:

    python project2_util.py
    
To get interactive documentation of the module, use `pydoc`:

    pydoc project2_util

For each example below, make sure you are in the `scratch` directory
in your terminal or command prompt before you begin.

The `scratch` directory has been set to be ignored by git when scanning
for new files to add. That way you can use it as a workspace for
work in progress that you're not ready to commit or push yet.


## Example 1:

Let's start with alpha-blending two images to produce a result:

    python ../project2_util.py blend ../example1/apple.jpg ../example1/orange.jpg ../example1/split_mask.png example1a.jpg
    
Note that the alpha mask itself was created with the `project2_util.py` script.
Try creating one yourself:

    python ../project2_util.py divide ../example1/apple.jpg my_first_mask.png
    
## Example 2:

The `project2_util.py` file can deal with regions of interest (ROI). I've
already created a few for the example images, but you can create your
own. Here's a command to create an ROI for Katherine Johnson's face.
When the window comes up, click the center of the left eye, then the
right eye, then the mouth.

    python ../project2_util.py select ../example2/johnson.jpg johnson.json

The ROI is saved to a json file in the `scratch` directory.

Once you have an ROI, you can show it:

    python ../project2_util.py show ../example2/henson.json

Or rotate and crop it (note this creates a new ROI in the `scratch`
folder for the cropped image):

    python ../project2_util.py crop ../example2/henson.json 2.2 3.0 1000 0 50 henson_crop.jpg henson_crop.json

You can also warp one image to align to the ROI of another image:

    python ../project2_util.py warp ../example2/johnson.json henson_crop.json johnson_crop.jpg 

If you want to verify alignment between two images, you can use the
`overlay` command:

    python ../project2_util.py overlay henson_crop.jpg johnson_crop.jpg 

## Example 3:

You can make elliptical masks using the `ellipse` command:

    python ../project2_util.py show ../example3/cushing.json
    python ../project2_util.py ellipse ../example3/cushing.json 1.4 2.2 cushing_mask.png

Let's warp another face to place it in the same place in the image (note use of `-flip` to flip horizontal as well):

    python ../project2_util.py show ../example3/prince.json
    python ../project2_util.py warp -flip ../example3/prince.json ../example3/cushing.json prince_warp.jpg
    
This is a lousy alpha blend of the two images:

    python ../project2_util.py blend ../example3/cushing.jpg prince_warp.jpg cushing_mask.png example3a.jpg

You will learn to make a higher-quality Laplacian pyramid blend for Project 2.

# Example 4:

This is a more ambitious "face swap". I probably could have spent more time defining & aligning my ROIs. 

Define first mask:

    python ../project2_util.py show ../example4/paul.json
    python ../project2_util.py ellipse ../example4/paul.json 1.7 2.3 paul_mask.png 
    
Define second mask:

    python ../project2_util.py show ../example4/elizabeth.json
    python ../project2_util.py ellipse ../example4/elizabeth.json 1.7 2.3 elizabeth_mask.png 
    
Warp Paul's face to where Elizabeth's is and vice versa (we again want to `-flip` because they are facing each other):

    python ../project2_util.py warp -flip ../example4/paul.json ../example4/elizabeth.json elizabeth_replacement.jpg
    python ../project2_util.py warp -flip ../example4/elizabeth.json ../example4/paul.json paul_replacement.jpg 

Combine the two result images above to get a single (badly-composited) image with the face locations swaped:
    
    python ../project2_util.py blend paul_replacement.jpg elizabeth_replacement.jpg ../example4/split_mask.png wandavision_faces_swapped.jpg
    
Combine the two mask images similarly:

    python ../project2_util.py blend paul_mask.png elizabeth_mask.png ../example4/split_mask.png wandavision_masks.png

Finally, combine original photo and the badly-composited face swap photo using the combined mask image:
    
    python ../project2_util.py blend ../example4/wandavision.jpg wandavision_faces_swapped.jpg wandavision_masks.png example4a.jpg

Again, the Laplacian pyramid blend you will implement will be higher quality than this alpha blend.

