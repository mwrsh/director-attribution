"""
This is the second step in our workflow.  In the folder containing this
script, be sure that you have a copy of deploy.prototxt.txt and
res10_300x300_ssd_iter_140000.caffemodel.

We point to a directory of subfolders, each of which represents a shot from our
target film as processed by Acut_detector.py.  This script will then loop
over each shot's frames, run facial detection on them, and do the following:

1. Determine the positions and sizes of each face detected in the frames,
then average out these values across the entire shot.
2. Determine the shot distance based on the ratio of the largest detected
face's height as compared to the height of the frame.
3. Record the shot distance and composition as a representative string that
is both human-readable and, eventually, statistically analysable.
4. Output these strings in a list in a .txt file.
5. (Optional) Average the pixel data of each frame in the shot, and then
draw over it the bounding boxes of the detected subjects' faces.  This is a
clumsy way to illustrate of what the compositional string actually
represents, albeit in a way that becomes quickly unreadable once a shot gets
long or complex enough!

This script uses OpenCV's built-in facial detection DNN, with the script
adapted from the invaluable work of Adrian Rosebrock at PyImageSearch.
As of March 2020, this is the highest-quality facial-detection method that
can be managed without a GPU in the volume of images we will be working
with.

I believe that users with the necessary hardware could get even better
results by using DVT's built-in MtCNN for face detection and VGGFace2 model
for face tracking via embedding comparison.
"""

import argparse
import cv2
import datetime
import os
import numpy as np

from math import sqrt

from PIL import Image, ImageDraw

from string import ascii_uppercase

execution_path = os.getcwd()

caffe_prototxt = os.path.join(execution_path, 'deploy.prototxt.txt')
caffe_model = os.path.join(execution_path,
                           'res10_300x300_ssd_iter_140000.caffemodel')
face_conf_threshold = 0.5
net = cv2.dnn.readNetFromCaffe(caffe_prototxt, caffe_model)

st_dt_obj = datetime.datetime.now()  # Start time.

shot_ratios = [0.90, 0.65, 0.45, 0.30, 0.15]
shot_names = ['XC', 'CU', 'MC', 'MS', 'ML']
"""
My own testing, as well as parallel testing by the DVT team, has shown that
the ratio of the largest face's bounding box to the height of the image
predicts shot distance, with the cut-offs above corresponding to XCU, CU,
MCU, M, and ML, respectively. More-distant shots cannot be consistently
tracked using face boxes.

We cannot use bounding boxes from object detection for these purposes, as the
dimensions of a person's bounding box are not invariant enough.

If this script is modified to use a different detection algorithm,
the ratios must be re-tested and modified.
"""

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--filepath', required=True,
                help='Path to the file directory.')
ap.add_argument('-n', '--name', required=True,
                help='Output filename.')
# ap.add_argument('-c', '--count',
#                help='Number of faces to track. Default is 5.')
ap.add_argument('-t', '--threshold',
                help='Frequency threshold of faces in the shot. Default is '
                     '20 percent.')
args = vars(ap.parse_args())

# if isinstance(args['count'], int):
#    max_count = args['count']
# else:
#    max_count = 5
max_count = 5
# I have included the option to track more than 5 subjects in each shot,
# but I have yet to encounter a meaningful number of films in which there
# are a significant number of shots with more than 5 detectable subjects.

full_file_path = str(args['filepath'])
comp_title = str(args['name'])
out_dir = os.path.join(full_file_path, 'output')

if isinstance(args['threshold'], float):
    freq_thresh = args['threshold']
else:
    freq_thresh = 0.1


def calc_distance(coord1, coord2):
    # Euclidean distance between two coordinates.
    dist = sqrt(((coord1[0] - coord2[0]) ** 2)
                + ((coord1[1] - coord2[1]) ** 2))
    return dist


def averaged_img(img_list):
    # Generate an averaged image from all images in the shot folder.  This is
    # just for illustrative purposes and is not actually necessary for the
    # analysis.
    w, h = Image.open(img_list[0]).size
    n = len(img_list)

    arr = np.zeros((h, w, 3), np.float)

    for im in img_list:
        imarr = np.array(Image.open(im), dtype=np.float)
        arr = arr + imarr / n

    arr = np.array(np.round(arr), dtype=np.uint8)
    return Image.fromarray(arr, mode='RGB')


def face_write(image):
    # Detects faces, finds the largest ones.  Returns a list of tuples of the
    # structure (height, width, (centroid_x ,centroid_y)) for later processing.

    # Much of the face detection work below is modified from Adrian
    # Rosebrock's work found at https://bit.ly/3btv1p5 ; all original credit
    # is due to Adrian, whose guides are truly invaluable.
    img_cv2 = cv2.imread(image)
    (img_h, img_w) = img_cv2.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(img_cv2, (300, 300)), 1.0,
                                 (300, 300), (104.0, 117.0, 123.0))
    # While some sources provide a BGR color triplet of (104, 177, 123) here
    # for the mean subtraction, this appears to be a widely repeated typo!
    # The green value must be 117, not 177.

    net.setInput(blob)
    dnn_detections = net.forward()

    face_list = []

    for i in range(0, dnn_detections.shape[2]):
        face_confidence = dnn_detections[0, 0, i, 2]

        if face_confidence > face_conf_threshold:
            box = dnn_detections[0, 0, i, 3:7] * np.array([img_w, img_h,
                                                          img_w, img_h])
            (startX, startY, endX, endY) = box.astype('int')
            face_h = endY - startY
            face_w = endX - startX
            face_cent = (int((endX + startX) / 2), int((endY + startY) / 2))
            face_data_tup = (face_h, face_w, face_cent)
            face_list.append(face_data_tup)

    if len(face_list) == 0:
        # Skip the rest if we did not detect any faces.
        return face_list

    else:
        # Sort the list in descending order of height of the bounding boxes.
        face_sorted = sorted(face_list, reverse=True)

        top_faces = []

        subject_count = 1

        for face_sort in face_sorted:
            # We now take as many subjects as we
            top_faces.append(face_sort)
            subject_count += 1
            if subject_count > max_count:
                break

        return top_faces


def subdirs(path):
    # Yield directory names not starting with '.' under given path.
    for subdir in os.scandir(path):
        if not subdir.name.startswith('.') and subdir.is_dir():
            yield subdir.name


def poi_gen9(image, out_list):
    # Function to generate a list of 9 keypoints reflecting the traditional
    # rule of thirds of composition.
    img_cv2 = image
    (img_h, img_w) = img_cv2.shape[:2]

    '''
    We need points of interest corresponding to 9 key points in a frame:

    .............|.............|.............
    ......1......|......2......|......3......
    .............|.............|.............
    -------------+-------------+-------------
    .............|.............|.............
    ......4......|......5......|......6......
    .............|.............|.............
    -------------+-------------+-------------
    .............|.............|.............
    ......7......|......8......|......9......
    .............|.............|.............

    If a face centroid's nearest keypoint is the one in section 4, then we can
    treat face as occupying the middle-left of the frame, and so on.
    '''

    w_6 = int(img_w / 6)
    h_6 = int(img_h / 6)

    points_of_int = [
        (w_6, h_6),  # 1
        ((3 * w_6), h_6),  # 2
        ((5 * w_6), h_6),  # 3
        (w_6, (3 * h_6)),  # 4
        ((3 * w_6), (3 * h_6)),  # 5
        ((5 * w_6), (3 * h_6)),  # 6
        (w_6, (5 * h_6)),  # 7
        ((3 * w_6), (5 * h_6)),  # 8
        ((5 * w_6), (5 * h_6))  # 9
    ]
    for point in points_of_int:
        out_list.append(point)


def nearest_coord(calc_coords, coord):
    # Function to find the nearest keypoint to each face centroid.
    dist_list = []
    for calc_coord in calc_coords:
        # Calculate distance, add to list the order of access.
        dist_list.append(calc_distance(calc_coord, coord))
    ind = dist_list.index(min(dist_list))
    return ind + 1


def nearest_val(calc_vals, val):
    # Function to find the nearest height ratio of the height of the largest
    # face.
    nearest = min(calc_vals, key=lambda x: abs(x - val))
    return calc_vals.index(nearest)


# Returns a list of the full filepaths of each subdirectory in the active
# folder.
shot_list = [os.path.join(full_file_path, shot) for shot in list(subdirs(
             full_file_path))]

total_shots = len(shot_list)

# We generate our output directory now, since otherwise it would've gotten
# caught up in the list comprehension above.
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

poi_list = []
comp_str_list = []
txt_file_name = '{}.txt'.format(comp_title)

# Gathering initial dimensions.
test_open = os.listdir(shot_list[0])
for frame in test_open:
    if not frame.startswith('.'):
        if frame.endswith('.png'):
            dims_img = cv2.imread(os.path.join(shot_list[0], frame))
            frame_h, frame_w = dims_img.shape[:2]
            poi_gen9(dims_img, poi_list)
            if len(poi_list) == 9:
                break

for shot_idx, shot in enumerate(shot_list):
    file_name = os.path.basename(os.path.normpath(shot))
    print('Working on shot {} of {}.'.format((shot_idx + 1), total_shots))

    # Collect all .png files from the folder
    frame_list = []
    for file in os.scandir(shot):
        if not file.name.startswith('.') and file.is_file() and \
                file.name.endswith('.png'):
            frame_list.append((os.path.join(shot, file)))

    # Generate a list of tuples for each shot containing information on
    # height, width, and centroid position for each face detected.
    faces_list = [face_write(frame) for frame in frame_list]

    # We want to create some buckets, into which we will average the size
    # and centroid data for the largest face, 2nd-largest face, 3rd-largest
    # face ... While this is not perfect, it gives us a rough indication of
    # the "average" composition of a shot.
    collect_list = [[] for inc in range(max_count)]
    avg_list = []
    for faces in faces_list:
        if len(faces) > 0:
            for face_idx, face_data in enumerate(faces):
                collect_list[face_idx].append(face_data)
                # We throw the face size data into each bucket appropriate
                # to its relative size in the image.  So, the 2nd-largest
                # image in the frame goes to the 2nd bucket.  That is not
                # perfect in terms of subject tracking, but it's still
                # mostly accurate when it comes to most real world examples.
                # Greater accuracy could be achieved using DVT's existing
                # VGGFace2 integration.
    len_list = [len(collect) for collect in collect_list]
    high_pop = max(len_list)
    # The face with the longest len_list is the face that appears in the
    # plurality of frames of the shot.  We judge whether to include a face
    # based on if it meets the frequency threshold as compared to the
    # high_pop face.
    if high_pop == 0:
        continue

    for collect in collect_list:
        if (len(collect) // high_pop) < freq_thresh:
            # If a face really only appears in a small number of frames,
            # we should skip it.  It's likely that it is either an error or
            # an insignificant character.
            continue
        else:
            total_h = 0
            total_w = 0
            total_centX = 0
            total_centY = 0

            for face_data in collect:
                total_h += face_data[0]
                total_w += face_data[1]
                total_centX += face_data[2][0]
                total_centY += face_data[2][1]

            avg_h = total_h // len(collect)
            avg_w = total_w // len(collect)
            avg_centX = total_centX // len(collect)
            avg_centY = total_centY // len(collect)
            avg_list.append((avg_h, avg_w, (avg_centX, avg_centY)))

    # At this point, we now have a list of tuples where each tuple reflects
    # the average size and position of the [N] most-prominent figures in the
    # shot.

    if len(avg_list) == 0:
        print('No faces detected. Skipping.')
        continue

    main_subj = avg_list[0]
    main_ratio = main_subj[0] / frame_h
    neighbor = nearest_val(shot_ratios, main_ratio)
    dist_name = shot_names[neighbor]
    # We now have the part of our string that would identify the shot
    # distance.

    # Now, we construct a partial string that will assign a letter to each
    # subject in the frame, along with the sector in which the face is
    # located.
    fill_count = 0
    letter_count = 0
    string_list = []
    position_list = []

    for face in avg_list:
        position = nearest_coord(poi_list, face[2])
        pos_2dig = str(position).zfill(2)
        pos_2dig = pos_2dig.replace('0', 'X')
        pos_2dig = pos_2dig.replace('1', 'A')
        pos_2dig = pos_2dig.replace('2', 'B')
        pos_2dig = pos_2dig.replace('3', 'C')
        pos_2dig = pos_2dig.replace('4', 'D')
        pos_2dig = pos_2dig.replace('5', 'E')
        pos_2dig = pos_2dig.replace('6', 'F')
        pos_2dig = pos_2dig.replace('7', 'G')
        pos_2dig = pos_2dig.replace('8', 'H')
        pos_2dig = pos_2dig.replace('9', 'I')
        # We cannot process the strings as 'words' without replacing their
        # numerical parts with letters.  I still want to keep them numerical up
        # until this point, because I believe there may be some use in the
        # future of treating positioning as something that may be treated
        # arithmetically.

        position_list.append(pos_2dig)

    while len(position_list) < max_count:
        # Padding for composition string, in case too few subjects found.
        position_list.append('nn')
        fill_count += 1

    for entry in position_list:
        string_list.append(ascii_uppercase[letter_count] + entry)
        letter_count += 1

    string_comb = ''.join(string_list)

    comp_str = '{}_Dist{}{}'.format(file_name, dist_name, string_comb)

    # Drawing section start.  Feel free to comment this out if you do not
    # need the illustrative averaged images.
    comp_file = os.path.join(out_dir, ('{}.png'.format(comp_str)))
    img_base = averaged_img(frame_list)
    draw_img = ImageDraw.Draw(img_base)

    for pt in poi_list:
        circ = ((pt[0] - 1), (pt[1] - 1), (pt[0] + 1), (pt[1] + 1))
        draw_img.ellipse(circ, outline=(128, 128, 128))

    for face_dims in avg_list:
        # Calculate rectangle coordinates from centroid and width, height.
        x_start = int(face_dims[2][0] - int(0.5 * face_dims[1]))
        x_end = int(face_dims[2][0] + int(0.5 * face_dims[1]))
        y_start = int(face_dims[2][1] - int(0.5 * face_dims[0]))
        y_end = int(face_dims[2][1] + int(0.5 * face_dims[0]))

        centroid = (int(face_dims[2][0]), int(face_dims[2][1]))
        rect_s = (x_start, y_start)
        rect_e = (x_end, y_end)
        circ = ((centroid[0] - 1), (centroid[1] - 1), (centroid[0] + 1),
                (centroid[1] + 1))

        draw_img.rectangle((rect_s, rect_e), outline=(255, 255, 0))
        draw_img.ellipse(circ, outline=(255, 255, 0))

    # Redraw the largest face in a different color, over everything else.
    large_face = avg_list[0]
    x_start = int(large_face[2][0] - int(0.5 * large_face[1]))
    x_end = int(large_face[2][0] + int(0.5 * large_face[1]))
    y_start = int(large_face[2][1] - int(0.5 * large_face[0]))
    y_end = int(large_face[2][1] + int(0.5 * large_face[0]))

    centroid = (int(large_face[2][0]), int(large_face[2][1]))
    rect_s = (x_start, y_start)
    rect_e = (x_end, y_end)

    circ = ((centroid[0] - 1), (centroid[1] - 1), (centroid[0] + 1),
            (centroid[1] + 1))

    draw_img.rectangle((rect_s, rect_e), outline=(255, 0, 0))
    draw_img.ellipse(circ, outline=(255, 0, 0))

    img_base.save(comp_file)
    # Drawing section end.

    comp_str_list.append(comp_str)

comp_str_file = os.path.join(out_dir, txt_file_name)

with open(comp_str_file, 'w') as out_file:
    for idx, composition in enumerate(comp_str_list):
        if idx + 1 < len(comp_str_list):
            out_file.write(composition)
            out_file.write('\n')
        else:
            out_file.write(composition)
