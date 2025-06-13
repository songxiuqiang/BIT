import numpy as np

import yaml
import cv2
import os

import communication

def LoadConfig(config_file):
    with open(config_file, 'r') as file:
        data = yaml.safe_load(file)
        return data

def LoadConfigSafety(config_file):
    with open(config_file, 'r') as file:
        lines = file.readlines()[1:]
        with open('temp_file.yml', 'w') as new_file:
            new_file.write("".join(lines))

    with open('temp_file.yml', 'r') as file:
        data = yaml.safe_load(file)

    os.remove('temp_file.yml')

    return data

def GenerateArray(first_num = 4, second_num = 5, len = 10):
    result = [first_num]
    for i in range(1, len):
        result.append(second_num)
    return result


def scale_cv_roi(cv_roi, scale = 1.0):
    if scale == 1.0:
        return cv_roi
    x, y, w, h = cv_roi[0], cv_roi[1], cv_roi[2], cv_roi[3]
    x_c = x + 0.5 * w  
    y_c = y + 0.5 * h
    _x = (int)(x_c - 0.5 * w * scale)  
    _y = (int)(y_c - 0.5 * h * scale)  
    _w = (int)(w * scale)
    _h = (int)(h * scale)
    return (_x, _y, _w, _h)


# convertion between OpenCV bbx and SAM BBX
def ConvertingBBX(bbx, cv2sam = False, sam2cv= False):
    if cv2sam and sam2cv:
        print("Illegal Condition!")
        return None
    res = bbx.copy()
    if cv2sam == True:
        res[2] = bbx[0] + bbx[2]
        res[3] = bbx[1] + bbx[3]
    if sam2cv == True:
        res[2] = bbx[2] - bbx[0]
        res[3] = bbx[3] - bbx[1]
    return res

def cv_roi_2_sam_bbx(cv_roi):
    res = cv_roi.copy()
    res[2] = cv_roi[0] + cv_roi[2]
    res[3] = cv_roi[1] + cv_roi[3]
    return res

def sam_bbx_2_cv_roi(sam_bbx):
    res = sam_bbx.copy()
    res[2] = sam_bbx[2] - sam_bbx[0]
    res[3] = sam_bbx[3] - sam_bbx[1]
    return res

# This function accepts a bounding box (bbx) and an offset value, and returns the center point and four corner coordinates of the bounding box with the specified offset.
# Parameters:
#   cvBBX (list): x, y, w, h, the type used in OpenCV
#   shift (float): the inward offset to be applied to the bounding box (default is 0.1).
# Returns:
#   list: a list containing the coordinates of the center point and the four corners of the shifted bounding box.
def pts_prompt_from_roi(cv_roi, shift = 0.1):
    x, y, w, h = cv_roi[0], cv_roi[1], cv_roi[2], cv_roi[3]
    x_shift = w * shift
    y_shift = h * shift
    c = (x + 0.5 * w, y + 0.5 * h)
    lt = (x + x_shift, y + y_shift)
    lb = (x + x_shift, y + h - y_shift)
    rt = (x + w - x_shift, y + y_shift)
    rb = (x + w - x_shift, y + h - y_shift)
    return [[c[0], c[1]], 
            [lt[0], lt[1]], 
            [lb[0], lb[1]], 
            [rt[0], rt[1]], 
            [rb[0], rb[1]]]


def ResizeMask(mask, origin_roi, target_roi, target_size, final_size, index = 0):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1)
    color = (30, 144, 255)
    mask_image = (mask_image * np.array(color)).astype(np.uint8)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    
    target_img = np.zeros((target_size[1], target_size[0], 1), dtype=np.uint8)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR) 
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY) 
    target_img[target_roi[1]:target_roi[1]+target_roi[3], target_roi[0]:target_roi[0]+target_roi[2]] = mask_image[origin_roi[1]:origin_roi[1]+origin_roi[3], origin_roi[0]:origin_roi[0]+origin_roi[2]]

    target_img = cv2.resize(target_img, (final_size[0],final_size[1]))
    # cv2.imshow("resized mask", target_img)
    return target_img

def InvertColor(imgs):
    invertered_imgs= []
    for img in imgs:
        invertered = 255 - img
        invertered_imgs.append(invertered)
    return invertered_imgs
    
    
def show_prompt(type = "", bg = None, mask = None, bbx = None, points = None, labels = None, index = 0, config = None, waitkey = 0):
    save_dir = config['save']['dir']			   
    
    show_mask   = bool(config['show']['mask'])
    show_bbx    = bool(config['show']['bbx'])
    show_points = bool(config['show']['points'])
    save_mask   = bool(config['save']['mask'])
    save_bbx    = bool(config['save']['bbx'])
    save_points = bool(config['save']['mask'])

    if type == "mask" and (mask is not None) and (show_mask or save_mask):
        color = (30, 144, 255)
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1)
        mask_image = (mask_image * np.array(color)).astype(np.uint8)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        if show_mask:
            cv2.imshow("mask", mask_image)
        if save_mask:
            cv2.imwrite(save_dir + 'mask/' + str(index) + '.png', mask_image)

    if type == "bbx" and (bbx is not None) and (show_bbx or save_bbx):
        x0, y0 = bbx[0], bbx[1]
        w, h = bbx[2] - bbx[0], bbx[3] - bbx[1]
        cv2.rectangle(bg, (int(x0), int(y0)), (int(x0+w), int(y0+h)), (0, 255, 0), 2)
        if show_bbx:
            cv2.imshow("bbx ", bg)
        if(save_bbx):
            cv2.imwrite(save_dir + 'bbx/' + str(index) + '.png', bg)

    if type == "points" and (points is not None) and (labels is not None) and (show_points or save_points):
        marker_size = 30
        pos_points = points[labels==1]
        neg_points = points[labels==0]
        for point in pos_points:
            cv2.drawMarker(bg, (int(point[0]), int(point[1])), (0, 255, 0), cv2.MARKER_STAR,
                       markerSize=marker_size, thickness=2, line_type=cv2.LINE_AA) 
        for point in neg_points:
            cv2.drawMarker(bg, (int(point[0]), int(point[1])), (0, 0, 255), cv2.MARKER_STAR,
                       markerSize=marker_size, thickness=2, line_type=cv2.LINE_AA)
        if show_points:
            cv2.imshow("points", bg)
        if save_points:
            cv2.imwrite(save_dir + 'points/' + str(index) + '.png', bg)
    
    # cv2.waitKey(waitkey)

def BinaringImgs(imgs, low_thresh = 127, high_thresh = 255):
    targets = []
    for img in imgs:
        _, target = cv2.threshold(img, low_thresh, high_thresh, cv2.THRESH_BINARY)
        targets.append(img)
        # cv2.imshow('target bin', target)
    return targets
        
def MultiplySamWithProb(sams, probs):
    targets = []
    for i in range(0,len(sams)):
        target = sams[i] * probs[i]
        targets.append(target)
        cv2.imshow("--sam",sams[i])
        cv2.imshow("--prob",probs[i])
        cv2.imshow("--target",target)
    return targets


def segment(predictor, img_width, img_height, dataGroups, configs, save_index):
    predict_masks = dataGroups.probs_.copy()
    print("len(dataGroups.rgbs_) ", len(dataGroups.rgbs_))
    for i in range(0, len(dataGroups.rgbs_)):
        cv_roi = scale_cv_roi(cv_roi = dataGroups.origin_rois_[i], scale = 1)
        bbx_prompt = np.array(cv_roi_2_sam_bbx(cv_roi = cv_roi))
        pts_prompt = np.array(pts_prompt_from_roi(cv_roi = cv_roi))
        label_prompt = np.array([1,0,0,0,0])
    
        predictor.set_image(dataGroups.rgbs_[i].copy())
        masks, scores, logits = predictor.predict(
	        box = bbx_prompt[None, :],
	        multimask_output = True,
        )
        bg = dataGroups.rgbs_[i].copy()
        show_prompt(type="mask", bg=bg, mask=masks[0], index=save_index, config=configs, waitkey=30)
        show_prompt(type="bbx", bg=bg, bbx=bbx_prompt, index=save_index, config=configs, waitkey=30)
        show_prompt(type="points", bg=bg, points=pts_prompt, labels=label_prompt, index=save_index, config=configs, waitkey=30)
		
        predict_masks[i] = ResizeMask(masks[0], dataGroups.origin_rois_[i], dataGroups.target_rois_[i], \
                                                    dataGroups.target_sizes_[i], (img_width,img_height), index = save_index)
        save_index = save_index + 1 
    return predict_masks