import os
from collections import namedtuple
import json

import cv2
import numpy as np
import sys

MAX_DISPLAY_W = 1200 
MAX_DISPLAY_H = 700

FIRST_IMSHOW = True

######################################################################

def get_display_scale(img):
    """Get integer scaling factor to fit image on screen."""
    h, w = img.shape[:2]
    return max(1, int(np.ceil(max(h / MAX_DISPLAY_H, w / MAX_DISPLAY_W))))

######################################################################

def scale_for_display(img, return_scale=False):

    """Downscale image for display on the screen. Returns scale factor and
    reduced image.""" 

    scl = get_display_scale(img)
    if scl == 1:
        display = img.copy()
    else:
        f = 1.0/scl
        display = cv2.resize(img, (0, 0), fx=f, fy=f,
                             interpolation=cv2.INTER_AREA)

    if return_scale:
        return display, scl
    else:
        return display

######################################################################

def put_text(display, prompt, loc=None):
    """Draw white-on-black text."""
    if loc is None:
        h = display.shape[0]
        loc = (4, h-8)

    for (brt, thk) in [(0, 3), (255, 1)]:
        cv2.putText(display, prompt, loc,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (brt, brt, brt), thk,
                    cv2.LINE_AA)

######################################################################

def get_single_click(window_name, img, prompt, default=None):
    """Get a single click location on an image. ESC quits program."""
    
    display, scl = scale_for_display(img, return_scale=True)

    put_text(display, prompt)

    mouse_pos = None

    def on_mouse(event, x, y, flags, param):
        nonlocal mouse_pos
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_pos = (x, y)

    cv2.setMouseCallback(window_name, on_mouse, None)

    cv2.imshow(window_name, display)

    global FIRST_IMSHOW
    if FIRST_IMSHOW:
        FIRST_IMSHOW = False
        cv2.moveWindow(window_name, 0, 0)

    while mouse_pos is None:
        k = cv2.waitKey(5)
        if k == 27:
            sys.exit(0)
        elif (default is not None) and (k == 10 or k == 13 ):
            mouse_pos = default

    cv2.setMouseCallback(window_name, lambda e, x, y, f, p: None, None)

    xy = np.ceil((np.array(mouse_pos, dtype=float) - 0.5) * scl).astype(int)

    return int(xy[0]), int(xy[1])

######################################################################

def setup_window(window_name):

    """Create an OpenCV window with hidden GUI."""

    wflags = cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_AUTOSIZE

    cv2.namedWindow(window_name, wflags)


######################################################################

def label_and_wait_for_key(window_name, display, prompt):

    display = scale_for_display(display)

    put_text(display, prompt)

    cv2.imshow(window_name, display)

    global FIRST_IMSHOW
    if FIRST_IMSHOW:
        FIRST_IMSHOW = False
        cv2.moveWindow(window_name, 0, 0)

    while cv2.waitKey(5) < 0: 
        pass


######################################################################

_ImageROI = namedtuple(
    'ImageROI', 
    'image_filename, center, angle, width, height'
)

######################################################################

def roi_from_center_angle_dims(image_filename,
                               center, angle, width, height):

    center = (float(center[0]), float(center[1]))
    angle = float(angle)
    width = float(width)
    height = float(height)

    return _ImageROI(image_filename, center, angle, width, height)

######################################################################

def roi_from_points(image_filename, top_left, top_right, bottom):

    """Create an ImageROI struct from three points clicked by user.
    Returns a namedtuple with fields: 

      * image_filename: filename of image
      * center:         center of ROI rectangle as (float, float) tuple
      * angle:          angle of ROI rectangle in radians
      * width:          width of ROI rectangle
      * height:         height of ROI rectangle, also used as 
                        scaling factor for warps

    """

    p0 = np.array(top_left, dtype=np.float32)
    p1 = np.array(top_right, dtype=np.float32)
    p2 = np.array(bottom)

    u = p1-p0
    width = np.linalg.norm(u)
    u /= width

    v = p2-p0

    if u[0] * v[1] - u[1] * v[0] < 0:
        u = -u
        top_left, top_right = top_right, top_left


    v -= u * np.dot(u, v) 

    assert np.abs(np.dot(u, v)) < 1e-4

    height = np.linalg.norm(v)

    cx, cy = p0 + 0.5*u*width + 0.5*v
    angle = np.arctan2(u[1], u[0])

    return _ImageROI(image_filename, 
                     (float(cx), float(cy)), 
                     float(angle), float(width), float(height))

######################################################################

def roi_get_matrix(image_roi):

    """Get a 3x3 matrix mapping local object points (x, y) in the ROI to
    image points (u, v) according to the formulas:

       x' = image_roi.height * x
       y' = image_roi.height * y

       c  = cos(image_roi.angle)
       s  = sin(image_roi.angle)

       u  = c * x' - s * y' + image_roi.center[0]
       v  = s * x' + c * y' + image_roi.center[1]

    """

    c = np.cos(image_roi.angle)
    s = np.sin(image_roi.angle)
    
    tx, ty = image_roi.center

    h = image_roi.height

    return np.array([[c*h, -s*h, tx],
                     [s*h, c*h, ty],
                     [0, 0, 1]])

######################################################################

def roi_map_points(image_roi, opoints):

    """Map from local object points to image points using the matrix
    established by roi_get_matrix(). The opoints parameter should be an
    n-by-2 array of (x, y) object points. The return value is an
    n-by-2 array of (u, v) pixel locations in the image.

    """

    M = roi_get_matrix(image_roi)

    opoints = opoints.reshape(-1, 1, 2)

    ipoints = cv2.perspectiveTransform(opoints, M)

    return ipoints.reshape(-1, 2)

######################################################################

def roi_display(window_name, image_roi, image=None, prompt=None):

    """Display the ROI on the image for debugging."""

    if image is None:
        image = cv2.imread(image_roi.image_filename)

    opoints = np.array([
        [-0.5, -0.5],
        [ 0.5, -0.5],
        [ 0.5,  0.5],
        [-0.5,  0.5],
        [-0.2,  0.0],
        [ 0.2,  0.0],
        [ 0.0, -0.2],
        [ 0.0,  0.2],
        [ 0.0,  0.5]
    ]) * np.array([image_roi.width/image_roi.height, 1])

    ipoints = roi_map_points(image_roi, opoints).astype(int)

    display = image.copy()
    scl = get_display_scale(display)

    cv2.polylines(display, [ipoints[:4]], True, 
                  (255, 255, 0), scl, cv2.LINE_AA)


    for i in [0, 1, -1]:
        cv2.circle(display, tuple(ipoints[i]), 4*scl, 
                   (255, 255, 0), scl, cv2.LINE_AA)

    cv2.line(display, tuple(ipoints[4]), tuple(ipoints[5]), 
             (255, 255, 0), scl, cv2.LINE_AA)

    cv2.line(display, tuple(ipoints[6]), tuple(ipoints[7]), 
             (255, 255, 0), scl, cv2.LINE_AA)

    if prompt is None:
        prompt = 'Selected region (hit a key when done)'

    label_and_wait_for_key(window_name, display, prompt)

######################################################################

def roi_save(output_filename, image_roi):

    """Save an ROI out in JSON format."""

    d = image_roi._asdict()

    odir = os.path.dirname(output_filename)

    d['image_filename'] = os.path.relpath(d['image_filename'], odir)

    with open(output_filename, 'w') as ostr:
        json.dump(d, ostr, indent=2)
        ostr.write('\n')

    print('wrote', output_filename)

######################################################################

def roi_load(input_filename):

    """Load an ROI from JSON format."""

    with open(input_filename, 'r') as istr:
        d = json.load(istr)

    idir = os.path.dirname(input_filename)
    d['image_filename'] = os.path.join(idir, d['image_filename'])

    for key in d:
        if isinstance(d[key], list):
            d[key] = tuple(d[key])

    return _ImageROI(**d)

######################################################################

def roi_warp(src_roi, dst_roi, src_image=None, dst_size=None, flip=False):

    """Warps the src_image so that its ROI overlaps the corresponding ROI
    in the destination image. Image scaling is based on height.
    """

    if src_image is None:
        src_image = cv2.imread(src_roi.image_filename)

    if dst_size is None:
        dst_image = cv2.imread(dst_roi.image_filename)
        h, w = dst_image.shape[:2]
        dst_size = (w, h)

    src_mat = roi_get_matrix(src_roi)
    dst_mat = roi_get_matrix(dst_roi)

    if flip:
        flip = np.diag([-1, 1, 1])
    else:
        flip = np.eye(3)

    M = dst_mat @ flip @ np.linalg.inv(src_mat) 

    return cv2.warpAffine(src_image, M[:2], dst_size, 
                          flags=cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_REFLECT_101)

######################################################################

def roi_draw_ellipse(img_roi, wsz, hsz, size=None): 

    """Draw an ellipse into an 8-bit single-channel mask image centered
    on the given ROI and rotated to align with it. The given dimensions
    are as fractions of the total height of the original ROI.
    """

    if size is None:
        src_image = cv2.imread(img_roi.image_filename)
        h, w = src_image.shape[:2]
    else:
        w, h = size

    mask = np.zeros((h, w), dtype=np.uint8)
    
    axes = 0.5 * img_roi.height * np.array([wsz, hsz])

    center = tuple([int(x) for x in img_roi.center])
    axes = tuple([int(x) for x in axes])

    deg = 180/np.pi

    return cv2.ellipse(mask, center, axes,
                       img_roi.angle*deg, 0, 360,
                       (255, 255, 255), -1, cv2.LINE_AA)

######################################################################

def select_interactive(window, *args):
    
    """Interactively select ROI's in one or more images. Parameters:

      * window is the name of an OpenCV window to tag in
    
      * remaining arguments should be one or more 
        (image_filename, roi_filename) pairs

    """
    
    for (image_filename, roi_filename) in args:

        #image_filename = os.path.abspath(image_filename)
        image = cv2.imread(image_filename)

        dest, _ = os.path.splitext(roi_filename)
        
        if image is None:
            raise RuntimeError('error reading ' + image_filename)
        
        scl = get_display_scale(image)

        top_left = get_single_click(window, image, 
                                    f'Click top left point for {dest}')

        display = image.copy()

        cv2.circle(display, top_left, 4*scl, 
                   (255, 255, 0), scl, cv2.LINE_AA)

        top_right = get_single_click(window, display,
                                     f'Click top right point for {dest}')

        cv2.circle(display, top_right, 4*scl, 
                   (255, 255, 0), scl, cv2.LINE_AA)

        cv2.line(display, top_left, top_right, 
                 (255, 255, 0), scl, cv2.LINE_AA)
        
        bottom = get_single_click(window, display,
                                  f'Click bottom point for {dest}')

        image_roi = roi_from_points(image_filename, 
                                    top_left, top_right, bottom)

        roi_save(roi_filename, image_roi)

        roi_display(window, image_roi, display)

######################################################################

def get_split_mask(size, point1, point2):

    """Make a single-channel 8-bit mask of the given size=(w, h) that
    splits the image in half along the line from point1 to point2. All
    points to the right of the line are white in the mask, and all
    points to the left of the line are black.

    """

    w, h = size
    k = 2*(w + h)

    p1 = np.array(point1, dtype=np.float32)
    p2 = np.array(point2, dtype=np.float32)
    
    pmid = 0.5*(p1 + p2)

    u = p2 - p1
    u /= np.linalg.norm(u)

    v = np.array([-u[1], u[0]])
    
    points = np.array([
        pmid + k * u,
        pmid + k * u + 2*k * v,
        pmid - k * u + 2*k * v,
        pmid - k * u
    ])

    points = (points + 0.5).astype(int).reshape(-1, 1, 2)

    mask = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(mask, [points], (255, 255, 255), cv2.LINE_AA)

    return mask

######################################################################

def alpha_blend(img1, img2, mask):
    
    """Perform alpha blend of img1 and img2 using mask.

    Result is an image of same shape as img1 and img2.  Wherever mask
    is 0, result pixel is same as img1. Wherever mask is 255 (or 1.0
    for float mask), result pixel is same as img2. For values in between,
    mask acts as a weight for a weighted average of img1 and img2.

    See https://en.wikipedia.org/wiki/Alpha_compositing
    """

    (h, w) = img1.shape[:2]

    assert img2.shape == img1.shape
    assert mask.shape == img1.shape or mask.shape == (h, w)

    result = np.empty_like(img1)

    if mask.dtype == np.uint8:
        mask = mask.astype(np.float32) / 255.0

    if len(mask.shape) == 2 and len(img1.shape) == 3:
        mask = mask[:, :, None]

    result[:] = img1 * (1 - mask) + img2 * mask

    return result

######################################################################

def draw_image_with_mask(image, mask):

    """Return a copy of image with the mask overlaid for display."""

    assert image.shape[:2] == mask.shape

    return alpha_blend(image // 2, image // 2 + 128, mask)

######################################################################

def visualize_pyramid(lp, padding=8):

    """Utility function to display a Laplacian pyramid."""

    n = len(lp)-1
    outputs = []

    h, w = lp[0].shape[:2]

    hmax = max([li.shape[0] for li in lp])

    hstackme = []

    hpadding = np.full((hmax, padding, 3), 255, np.uint8)

    for i, li in enumerate(lp):

        assert li.dtype == np.float32

        if i == n:
            display = li
        else:
            display = 127 + li

        display = np.clip(display, 0, 255).astype(np.uint8)

        h, w = display.shape[:2]

        if h < hmax:
            vpadding = np.full((hmax - h, w, 3), 255, np.uint8)
            display = np.vstack((display, vpadding))


        if i > 0:
            hstackme.append(hpadding)

        hstackme.append(display)

    return np.hstack(tuple(hstackme))

######################################################################

def _dieusage():
    print('usage: python project2_util.py select IMAGE1 OUTPUT1.json [IMAGE2 OUTPUT2.json ...]')
    print('   or: python project2_util.py show IMAGE_ROI.json')
    print('   or: python project2_util.py crop [-flip] SOURCE.json WSZ HSZ HEIGHT RESULT.jpg [RESULT.json]')
    print('   or: python project2_util.py warp [-flip] SOURCE.json DEST.json RESULTIMG.jpg [RESULT.json]')
    print('   or: python project2_util.py ellipse SOURCE.json WSZ HSZ MASK.png')
    print('   or: python project2_util.py divide IMAGE1 MASK.png')
    print('   or: python project2_util.py overlay IMAGE1 IMAGE2 OUTPUT')
    print('   or: python project2_util.py blend IMAGE1 IMAGE2 MASK OUTPUT')
    print()
    sys.exit(1)

######################################################################

def _select_command(args):

    if len(args) % 2 != 0:
        _dieusage()

    inputs = []
    
    for i in range(0, len(args), 2):
        image_filename = args[i]
        roi_filename = args[i+1]
        inputs.append((image_filename, roi_filename))

    window = 'Select region'
    setup_window(window)

    select_interactive(window, *inputs)

######################################################################

def _show_command(args):

    window = 'Selected region'
    setup_window(window)

    for tagfile in args:

        image_roi = roi_load(tagfile)

        roi_display(window, image_roi)

######################################################################

def _warp_command(args):

    if len(args) < 3 or len(args) > 5:
        _dieusage()

    flip = False

    if args[0] == '-flip':
        flip = True
        args = args[1:]

    src_roi = roi_load(args[0])
    dst_roi = roi_load(args[1])
    result_filename = args[2]

    warped = roi_warp(src_roi, dst_roi, flip=flip)
    cv2.imwrite(result_filename, warped)
    print('wrote', result_filename)

    dst_roi = roi_from_center_angle_dims(result_filename,
                                         dst_roi.center,
                                         dst_roi.angle,
                                         src_roi.width * dst_roi.height / src_roi.height,
                                         dst_roi.height)

    if len(args) == 4:
        roi_save(args[3], dst_roi)

    window = 'Warp result'
    setup_window(window)

    roi_display(window, dst_roi, warped, 
                'Warp result (hit a key when done)')

######################################################################

def _crop_command(args):

    if len(args) < 7 or len(args) > 9:
        _dieusage()

    flip = False
    
    if args[0] == '-flip':
        flip = True
        args = args[1:]
        
    src_roi = roi_load(args[0])
    
    wsz = float(args[1])
    hsz = float(args[2])
    result_height = int(args[3])
    scroll_x = int(args[4])
    scroll_y = int(args[5])

    result_filename = args[6]

    src_image = cv2.imread(src_roi.image_filename)

    wpx = wsz * src_roi.height
    hpx = hsz * src_roi.height

    result_width = int(round(result_height * wpx / hpx))

    scl = result_height / (src_roi.height * hsz)

    dst_roi = roi_from_center_angle_dims(result_filename,
                                         (0.5*result_width + scroll_x, 0.5*result_height + scroll_y),
                                         0.0,
                                         src_roi.width * scl,
                                         src_roi.height * scl)


    dst_size = (result_width, result_height)


    result_image = roi_warp(src_roi, dst_roi, src_image, dst_size, flip)
    cv2.imwrite(result_filename, result_image)

    if len(args) == 8:
        roi_save(args[7], dst_roi)

    window = 'Crop result'
    setup_window(window)

    roi_display(window, dst_roi, result_image, 
                'Crop result (hit a key when done)')

######################################################################

def _ellipse_command(args):

    if len(args) != 4:
        _dieusage()

    src = roi_load(args[0])
    wsz = float(args[1])
    hsz = float(args[2])
    result = args[3]

    src_image = cv2.imread(src.image_filename)
    h, w = src_image.shape[:2]
    src_size = (w, h)

    ellipse = roi_draw_ellipse(src, wsz, hsz, src_size)

    cv2.imwrite(result, ellipse)
    print('wrote', result)

    window = 'Ellipse result'
    setup_window(window)

    display = draw_image_with_mask(src_image, ellipse)

    label_and_wait_for_key(window, ellipse,
                           'Ellipse result (hit a key when done)')

    label_and_wait_for_key(window, display, 
                           'Overlaid on image (hit a key when done)')

######################################################################

def _divide_command(args):

    if len(args) != 2:
        _dieusage()

    image_filename = args[0]
    mask_filename = args[1]

    window = 'Divide image'
    setup_window(window)

    image = cv2.imread(image_filename)

    point1 = get_single_click(window, image, 
                                 'Click first point along dividing line')

    display = image.copy()
    scl = get_display_scale(display)

    cv2.circle(display, point1, 4*scl,
               (255, 255, 0), scl, cv2.LINE_AA)

    point2 = get_single_click(window, image,
                                 'Click second point along dividing line')

    h, w = image.shape[:2]

    mask = get_split_mask((w, h), point1, point2)

    cv2.imwrite(mask_filename, mask)
    print('wrote', mask_filename)

    display = draw_image_with_mask(image, mask)

    label_and_wait_for_key(window, mask,
                           'Divide result (hit a key when done)')

    label_and_wait_for_key(window, display, 
                           'Overlaid on image (hit a key when done)')
    
######################################################################

def _blend_command(args):

    if len(args) != 4:
        _dieusage()

    image1 = cv2.imread(args[0], cv2.IMREAD_ANYCOLOR)
    image2 = cv2.imread(args[1], cv2.IMREAD_ANYCOLOR)
    mask = cv2.imread(args[2], cv2.IMREAD_GRAYSCALE)
    result = args[3]

    blend = alpha_blend(image1, image2, mask)
    cv2.imwrite(result, blend)
    print('wrote', result)

    display = scale_for_display(blend)

    window = 'Blend result'
    setup_window(window)
    
    label_and_wait_for_key(window, display, 
                           'Blend result (hit a key when done)')

######################################################################

def _overlay_command(args):

    if len(args) != 2:
        _dieusage()

    image1 = cv2.imread(args[0], cv2.IMREAD_ANYCOLOR)
    image2 = cv2.imread(args[1], cv2.IMREAD_ANYCOLOR)

    blend = np.empty_like(image1)
    t = 0.0

    window = 'Overlay result'
    setup_window(window)

    while True:
        u = np.cos(t) * 0.5 + 0.5
        t += 0.05
        blend[:] = image1*(1.0 - u) + image2*u
        display = scale_for_display(blend)
        cv2.imshow(window, display)
        put_text(display, 'Overlay result (hit a key when done)')
        k = cv2.waitKey(5)
        if k >= 0:
            break
    
######################################################################

def _main():

    command_funcs = dict(
        select = _select_command,
        show = _show_command,
        crop = _crop_command,
        warp = _warp_command,
        ellipse = _ellipse_command,
        divide = _divide_command,
        overlay = _overlay_command,
        blend = _blend_command
    )
    
    if len(sys.argv) < 3 or sys.argv[1] not in command_funcs:
        _dieusage()

    func = command_funcs[sys.argv[1]]

    func(sys.argv[2:])

if __name__ == '__main__':
    _main()
