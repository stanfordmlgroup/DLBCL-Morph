import cv2
import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
import matlab.engine

from PIL import Image
from tqdm import tqdm
from scipy import stats
from scipy.special import ellipe
from scipy.ndimage import binary_fill_holes
from scipy.spatial.distance import squareform, pdist

# Read in single RBC patch
def read_image_npy(img_path):
    img = np.load(img_path)
    img = binary_fill_holes(img)
    mask = img.copy()
    img = img.astype(np.uint8)
    img[mask] = 255

    ret, thresh = cv2.threshold(img,127,255,0)
    _, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]

    return thresh, cnt

def read_image(img_dir):
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(img,127,255,0)
    _, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]

    return thresh, cnt

# Save the grayscale image
def save_binary_img(img_name, save_dir, thresh):
    img_name = img_name[:-4] + ".png"
    img_arry = np.zeros(thresh.shape)
    img_arry = (thresh == 255)
    img_arry=img_arry*1
    cv2.imwrite(os.path.join(save_dir, img_name), thresh)



# Calculate regional shape factors
def compute_ellipticalSF(thresh, cnt):
    # fit minimum bounded rectangle
    rect = cv2.minAreaRect(cnt)
    (rectCoord1, rectCoord2, rotate_angle) = rect

    # fit minimum bounded ellipse
#     ret,thresh = cv2.threshold(img,127,255,0)
    ellipse = cv2.fitEllipseDirect(cnt)  #(x, y), (MA, ma), angle
    ell = cv2.ellipse(thresh,ellipse,(169,169,169),3)
    (ellpCtr_x, ellpCtr_y), (shortAxis, longAxis), angle = ellipse

    # perimeter and area of ellipse
    a = longAxis / 2
    b = shortAxis / 2
    e = np.sqrt(1 - b**2 / a**2)  # eccentricity
    perimt = 4 * a * ellipe(e*e)
    area = np.pi * a * b

    return rectCoord1, rectCoord2, rotate_angle, (ellpCtr_x, ellpCtr_y), shortAxis, longAxis, perimt, area

# Calculate convex hull related shape factors
def compute_regionalSF(cnt):
    # fit convex hull
    hull = cv2.convexHull(cnt, returnPoints = False)

    defects = cv2.convexityDefects(cnt,hull)
    dist_hist = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        dist = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
        dist_hist.append(dist)
        far = tuple(cnt[f][0])

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hull_perimtr = np.sum(dist_hist)

    return hull_area, hull_perimtr

# Calculate min and max feret diameters
def compute_feretDiameter(binaryImg_dir, eng):
    # run Matlab engine
    binaryImg_dir = binaryImg_dir[:-4] + ".png"
    img = eng.imread(binaryImg_dir);
    os.remove(binaryImg_dir)
    # threshold = eng.graythresh(img);
    # img = eng.im2bw(img, threshold);
    bw = eng.imbinarize(img);
    bw = eng.imfill(bw,'holes');

    eng.workspace['resMin'] = eng.bwferet(bw, 'MinFeretProperties')
    eng.workspace['resMax'] = eng.bwferet(bw, 'MaxFeretProperties')

    resMin = eng.extract_res(eng.workspace['resMin'])
    resMin = eng.table2cell(resMin)
    resMax = eng.extract_res(eng.workspace['resMax'])
    resMax = eng.table2cell(resMax)

    (MinDiameter, MinAngle, MinCoordinates) = resMin
    (MaxDiameter, MaxAngle, MaxCoordinates) = resMax
    return MinDiameter, MaxDiameter, MinAngle, MaxAngle

def compute_derivedSF(shortAxis, longAxis, area, perimt, MinDiameter, MaxDiameter, hull_area, hull_perimtr):
    esf = shortAxis/longAxis
    csf = 4 * np.pi * area / perimt**2
    sf1 = shortAxis / MaxDiameter
    sf2 = MinDiameter / MaxDiameter
    elg = MaxDiameter / MinDiameter
    cvx = np.sqrt(area / hull_area)
    cmpt = 4 * np.pi * area / hull_perimtr

    return esf, csf, sf1, sf2, elg, cvx, cmpt

def main(image_dir, bimgsave_dir, frame_cutoff=0, eng=None):

    for i, date in enumerate(os.listdir(image_dir)):
        print(f'{i + 1}/{len(os.listdir(image_dir))}: {date}')
        date_dir = os.path.join(image_dir, date)
        if not os.path.isdir(date_dir) or date == '2019-05-10':
            continue

        binary_date = os.path.join(bimgsave_dir, date)
        if not os.path.exists(binary_date):
            os.makedirs(binary_date)

        for sample in tqdm(os.listdir(date_dir)):
            sample_dir = os.path.join(date_dir, sample, 'Patches')

            binary_sample = os.path.join(binary_date, sample)
            if not os.path.exists(binary_sample):
                os.makedirs(binary_sample)

            # new array for each sample
            cvx_list = []
            elg_list = []

            # directory to save .txt files
            save_dir = os.path.join(date_dir, sample)

            if os.path.exists(os.path.join(save_dir, 'elongation.txt')):
                continue

            # iterate through all last 50% frames for each sample
            for frame in os.listdir(sample_dir):
                frame_no = frame.split('_')[2]

                # initailize dataframe for each frame no.
                shape_factor_df = pd.DataFrame(columns=['patch_name', 'rectCoord1', 'rectCoord2', 'rotate_angle', 'ellip_centroid', \
                                            'shortAxis', 'longAxis', 'ellip_perimt', 'ellip_area', 'hull_area', \
                                            'hull_perimtr', 'minDiameter', 'maxDiameter', 'minAngle', 'maxAngle',\
                                            'esf', 'csf', 'sf1', 'sf2', 'elogation', 'convexity', 'compactness'])

                if int(frame_no) >= frame_cutoff:
                    binary_frame = os.path.join(binary_sample, frame)
                    if not os.path.exists(binary_frame):
                        os.makedirs(binary_frame)

                    patch_dir = os.path.join(sample_dir, frame)
                    for patch in os.listdir(patch_dir):
                        if patch.endswith(".npy"):
                            thresh, cnt = read_image_npy(os.path.join(patch_dir, patch))
                            save_binary_img(patch, binary_frame, thresh)
                            try:
                                rectCoord1, rectCoord2, rotate_angle, (ellpCtr_x, ellpCtr_y), \
                                        shortAxis, longAxis, perimt, area = compute_ellipticalSF(thresh, cnt)
                                hull_area, hull_perimtr = compute_regionalSF(cnt)
                                MinDiameter, MaxDiameter, MinAngle, MaxAngle = compute_feretDiameter(os.path.join(binary_frame, patch), eng)
                                esf, csf, sf1, sf2, elg, cvx, cmpt = compute_derivedSF(
                                    shortAxis, longAxis, area, perimt, MinDiameter,
                                    MaxDiameter, hull_area, hull_perimtr)
                            except Exception as inst:
                                # One of the excptions (the only one observed yet) is when the cell is cut off when patches where constructed
                                print(os.path.join(patch_dir, patch))
                                print(inst)
                                continue

                            # append shape factor to list
                            cvx_list.append(cvx)
                            elg_list.append(elg)

                            # shape_factor_df = shape_factor_df.append([[patch, rectCoord1, rectCoord2, rotate_angle, \
                            # 	(ellpCtr_x, ellpCtr_y), shortAxis, longAxis, perimt, area, hull_area, hull_perimtr, \
                            # 	MinDiameter, MinAngle, MaxDiameter, MaxAngle, esf, csf, sf1, sf2, elg, cvx, cmpt]])
                            shape_factor_df = shape_factor_df.append({'patch_name':patch, 'rectCoord1':rectCoord1, 'rectCoord2':rectCoord2, 'rotate_angle':rotate_angle, \
                                        'ellip_centroid':(ellpCtr_x, ellpCtr_y), 'shortAxis': shortAxis,'longAxis':longAxis, 'ellip_perimt':perimt, 'ellip_area':area, \
                                        'hull_area': hull_area,'hull_perimtr': hull_perimtr, 'minDiameter':MinDiameter, 'maxDiameter':MaxDiameter, 'minAngle':MinAngle, 'maxAngle':MaxAngle,\
                                        'esf':esf, 'csf':csf, 'sf1':sf1, 'sf2':sf2, 'elogation':elg, 'convexity':cvx, 'compactness':cmpt}, ignore_index=True)

                        output_dir = os.path.join(binary_frame, 'shape.csv')
                        shape_factor_df.to_csv(output_dir)

            # generate convexity.txt
            cvx_mean = np.mean(cvx_list)
            cvx_std = np.std(cvx_list)
            cvx_skew = stats.skew(cvx_list)
            cvx_kurtosis = stats.kurtosis(cvx_list)

            with open(os.path.join(save_dir, 'convexity.txt'), 'w') as f:
                f.write(f'Convexity Mean: {cvx_mean}\n')
                f.write(f'Convexity Standard Deviation: {cvx_std}\n')
                f.write(f'Convexity Skew: {cvx_skew}\n')
                f.write(f'Convexity Kurtosis: {cvx_kurtosis}\n')

            # generate elongation.txt
            elg_mean = np.mean(elg_list)
            elg_std = np.std(elg_list)
            elg_skew = stats.skew(elg_list)
            elg_kurtosis = stats.kurtosis(elg_list)

            with open(os.path.join(save_dir, 'elongation.txt'), 'w') as f:
                f.write(f'Elongation Mean: {elg_mean}\n')
                f.write(f'Elongation Standard Deviation: {elg_std}\n')
                f.write(f'Elongation Skew: {elg_skew}\n')
                f.write(f'Elongation Kurtosis: {elg_kurtosis}\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default='/deep/group/aihc-bootcamp-spring2019/blood/data/images')
    parser.add_argument("--bimgsave_dir", default='/deep/group/aihc-bootcamp-spring2019/blood/data/binary_image')
    # parser.add_argument("--output_dir", default='/Users/yanchengli/Desktop/patches_binary/output.csv')

    args = parser.parse_args()

    # start Matlab engine
    eng = matlab.engine.start_matlab()

    #read_image("/Users/damir/Documents/Stanford/ml-group/blood/blood_data/06-17/9/Patches/Frame 70/cell1.png")
    #read_image_npy("/Users/damir/Documents/Stanford/ml-group/blood/tst/blood_dir/06-17/patient_13986/Patches/patch_id_10173/cell_1.npy")
    main(args.image_dir, args.bimgsave_dir, eng=eng)
