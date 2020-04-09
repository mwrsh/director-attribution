"""
This script is the first step in our workflow.  Given a video file this
script will provide 5 60-second clips that are by default set to 1/6th,
2/6th, 3/6th ... of the way through the file.  The user then views the clips
and manually counts the number of cuts in it.

After the five clips have been manually analysed, the script runs through
potential parameter settings for each clip until it reaches the setting that
will get closest to the right number of shots.  The parameter results are
then averaged, and the DVT CutAggregator script is run on the full video,
using the averaged settings.

In extensive testing, averaging the parameters in this manner leads to
perfect or near-perfect cut detection.  This allows us to get almost perfect
results for shot detection without any actual manual experimentation with
settings, and requires us only to process 5 additional minutes' worth of
video, rather than repeatedly processing the same full video file while
testing different settings.

Upon detecting the cut-off points for the shots, while it would be nice to
just output individual video files for each shot for our next steps in the
workflow, we are instead going to output each scene as a directory
containing one screenshot for each 0.5 seconds of footage.  This will make
the next step in our workflow doable in a manageable timeframe for users
with lower-powered setups like, say, my 2018 MacBook Air.

After running this script, users should manually remove non-content shot
directories from the output folder (e.g. credits, previews, commercials, etc.).

Generally, when there are errors in the cut detection process, they will be
consistent enough to easily fix--by viewing each subdirectories' images as
thumbnails, it will be clear when the first few frames originate from
another clip.  It is fairly trivial to move these error frames into the
preceding subdirectory, either manually or via a helper script.
"""

import argparse
import cv2
import datetime
import multiprocessing as mp
import os

from PIL import Image
from dvt.aggregate.cut import CutAggregator
from dvt.annotate.diff import DiffAnnotator
from dvt.core import DataExtraction, FrameInput
from moviepy.editor import VideoFileClip

from statistics import mean


def manual_cut_count(man_cut_vid, out_list):
    man_cut_cap = cv2.VideoCapture(man_cut_vid)
    # w_width = 640
    # multi = w_width // man_cut_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # # To maintain ratio.
    # w_height = int(multi * man_cut_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    user_count = 0
    wk = int((1000 * (1 / frate)) / 2)  # Roughly 2x speed.

    if not man_cut_cap.isOpened():
        print('Error opening video.')

    cv2.namedWindow('Count the cuts.', cv2.WINDOW_NORMAL)
    # cv2.startWindowThread()
    # cv2.resizeWindow('Count the cuts.', (w_width, w_height))
    font = cv2.FONT_HERSHEY_SIMPLEX

    while man_cut_cap.isOpened():
        ret, frame = man_cut_cap.read()
        if ret:
            if frame is not None:
                # Press 'A' if there is a cut. You can use 'S' to undo
                # if necessary.
                txt = 'Shot count: {}'.format(user_count)
                cv2.putText(frame, txt, (20, 20), font, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
                cv2.imshow('Count the cuts.', frame)

                k = cv2.waitKey(wk) & 0xFF

                if k == ord('a'):
                    user_count += 1

                if k == ord('s'):
                    user_count -= 1

                if k == ord('q'):
                    break
        else:
            break

    cv2.destroyAllWindows()
    for _ in range(1, 10):
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    for _ in range(1, 10):
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    for _ in range(1, 10):
        cv2.waitKey(1)
    man_cut_cap.release()

    affirm = ['y', 'yea', 'yes', 'yeah', 'ye']
    negate = ['n', 'nah', 'nay', 'no', 'nope']
    possible = affirm + negate

    print('You counted {} shots.'.format(user_count))
    while True:
        correct = input('Is this correct? ').lower()
        if correct not in possible:
            print('Sorry, please answer \'yes\' or \'no\'.')
            continue
        else:
            if correct in affirm:
                print('Ok! Will test thresholds until match is found.')
                break
            else:
                print('Ok. Please modify your entry. ')
                while True:
                    try:
                        user_count = int(input('Enter number of shots. '))
                    except ValueError:
                        print('Please enter an integer value. ')
                        continue
                    else:
                        break

    if user_count < 2:
        user_count = 2
        # Things screw up if the script is told that the clip has less than
        # 2 shots in it.  I am unsure of why this is, but forcing it to a
        # minmum of 2

    out_list.append(user_count)


def dextra_diff(ddiff_vid):
    dextra = DataExtraction(FrameInput(input_path=ddiff_vid))
    dextra.run_annotators([DiffAnnotator(quantiles=[quant_thresh])])
    return dextra


def find_cuts(find_cut_extract, find_cut_val):
    cut_key = 'q{}'.format(quant_thresh)
    find_cut_extract.run_aggregator(CutAggregator(cut_vals={cut_key:
                                                            find_cut_val}))

    cut_frames = find_cut_extract.get_data()['cut']
    cut_fs = []
    cut_tups = []

    for index, row in cut_frames.iterrows():
        cut_tups.append((row['frame_start'], row['frame_end']))
        cut_fs.append(row['frame_start'])
        cut_fs.append(row['frame_end'])

    cut_count = len(cut_tups)
    while cut_count < 2:
        cut_count += 1
        # Testing has shown that the detector does not easily parse long shots,
        # and really wants to consider a single unbroken shot, no matter how
        # stable, to be two shots. Indulge it to avoid outliers.

    print('{} shots detected at {}.'.format(cut_count, find_cut_val))
    return cut_key, cut_count, cut_tups, cut_fs


num_processes = mp.cpu_count()

st_dt_obj = datetime.datetime.now()

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--filepath', required=True,
                help='Path to the file.')
ap.add_argument('-p', '--palette', choices=['c', 'b'], required=True,
                help='Enter C for color, B for black and white.')
ap.add_argument('-a', '--aseg', type=int,
                help='Time in seconds at which to start first test clip. \
                        default is 1/6th of the way through file.')
ap.add_argument('-b', '--bseg', type=int,
                help='Time in seconds at which to start second test clip. \
                        default is 1/3rd of the way through file.')
ap.add_argument('-c', '--cseg', type=int,
                help='Time in seconds at which to start third test clip. \
                        default is 1/2 of the way through file.')
ap.add_argument('-d', '--dseg', type=int,
                help='Time in seconds at which to start fourth test clip. \
                        default is 2/3rd of the way through file.')
ap.add_argument('-e', '--eseg', type=int,
                help='Time in seconds at which to start fifth test clip. \
                        default is 5/6th of the way through file.')
ap.add_argument('-l', '--length', type=int, default=60,
                help='Time in seconds for the test clips. Default is 60.')
args = vars(ap.parse_args())

inp_file = str(args['filepath'])
full_file_name = os.path.basename(inp_file)
full_file_dir = os.path.dirname(inp_file)
orig, ext = full_file_name.split('.')
full_film = VideoFileClip(inp_file)
full_copy = full_film.copy()

test_dir = os.path.join(full_file_dir, orig)
os.makedirs(test_dir, exist_ok=True)

# While DVT can grab framerate as well, this works quite a bit quicker for
# our purposes.
cap = cv2.VideoCapture(inp_file)
fr_wid = cv2.CAP_PROP_FRAME_WIDTH
fr_hgt = cv2.CAP_PROP_FRAME_HEIGHT
frate = cap.get(cv2.CAP_PROP_FPS)
f_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
s_length = f_length / frate
cap.release()

if args['palette'] == 'b':
    quant_thresh = 85
else:
    quant_thresh = 40

'''
A quantile threshold of 40% works well for color films.  A quantile
threshold of 85% is optimum for black and white films, according to DVT
co-author Dr. Taylor Arnold, because any two frames in a black and white
film, in the HSV space, already share 67% of their values, since the hue
and saturation are always the same.
https://github.com/distant-viewing/dvt/issues/24
'''

if isinstance(args['aseg'], int):
    aseg_start = int(args['aseg'])
else:
    aseg_start = int(s_length / 6)

if isinstance(args['bseg'], int):
    bseg_start = int(args['bseg'])
else:
    bseg_start = int(2 * (s_length / 6))

if isinstance(args['cseg'], int):
    cseg_start = int(args['cseg'])
else:
    cseg_start = int(3 * (s_length / 6))

if isinstance(args['dseg'], int):
    dseg_start = int(args['dseg'])
else:
    dseg_start = int(4 * (s_length / 6))

if isinstance(args['eseg'], int):
    eseg_start = int(args['eseg'])
else:
    eseg_start = int(5 * (s_length / 6))

seg_dur = int(args['length'])

clip_starts = [aseg_start, bseg_start, cseg_start, dseg_start, eseg_start]

time_stamps = []

test_clips = []

for start in clip_starts:
    tup = (start, (start + seg_dur))
    time_stamps.append(tup)

os.chdir(test_dir)

for idx, tup in enumerate(time_stamps):
    # For testing, save time by commenting out the lines starting with
    # 'test_seg' in this loop.
    test_seg = full_film.subclip(t_start=tup[0], t_end=tup[1])  # This one.
    seg_name = 'seg{}.mp4'.format((idx + 1))
    seg_path = os.path.join(test_dir, seg_name)
    print(seg_path)
    test_seg.write_videofile(seg_name, fps=frate, threads=4, audio=False,
                             codec=None, preset='ultrafast',
                             logger=None)  # And this one.
    test_clips.append(seg_path)

user_count_list = []
final_vals = []

for vid in test_clips:
    manual_cut_count(vid, user_count_list)

# For testing, comment out the 'for vid in test_clips' loop above and put your
# calculated counts in this list below.
# user_count_list = [4, 3, 7, 5, 3]

postcount_dt_obj = datetime.datetime.now()

for idx, clip in enumerate(test_clips):
    dex_obj = dextra_diff(clip)
    val = 0
    found = False
    dets = []

    while not found:
        det_key, det_cuts, det_tups, det_frames = find_cuts(dex_obj, val)
        if det_cuts == user_count_list[idx]:
            final_val = val
            final_vals.append(final_val)
            found = True
        elif det_cuts > user_count_list[idx]:
            '''
            If we have not yet found the best result so far (i.e. we just
            started), then we track the returned cuts, val tuple.  If instead
            we have a best result so far, we instead check to see if our new
            result is any better. If it is not, we change nothing.  If it is,
            we update it.  By doing this, we refer back to our best result
            if things go too far, as in the next elif.
            '''
            dets.append(det_cuts)
            val += 1
        elif det_cuts < user_count_list[idx]:
            if val == 0:
                final_val = val
            else:
                search = dets[-1]
                # Find the last detected cut count from before we went too low.
                final_val = (dets.index(search)) + 1
                # Return the index of the first val that gave us that result
                # and add 1.
            final_vals.append(final_val)
            found = True

full_val = mean(final_vals)

# For quicker testing, comment out everything from 'final_vals = []' to here,
# and put your pre-calculated full_val below:
# full_val = 0

dt_obj = datetime.datetime.now()
et_obj = (dt_obj - postcount_dt_obj).total_seconds()
print('Ideal value found at {}, in {}.'.format(dt_obj, et_obj))
print('Set value to {} for cutting the full film.'.format(full_val))

dex_obj = dextra_diff(inp_file)
fnl_key, fnl_cuts, fnl_tups, fnl_frames = find_cuts(dex_obj, full_val)

print(fnl_tups)


def fnl_tup_processing(g_num):
    fnl_tup_divide = len(fnl_tups) // num_processes
    g_start = g_num * fnl_tup_divide
    g_end = (g_num + 1) * fnl_tup_divide
    fnl_tups_sec = fnl_tups[g_start:g_end]
    # Divide the list of tuples into one section per core.

    for idx1, tup1 in enumerate(fnl_tups_sec):
        f_start = tup1[0] + 1
        f_end = tup1[1]
        f_inds = [i for i in range(f_start, f_end) if i % (frate // 2) == 0]
        # This returns just one frame every half second.

        # However, we want to always include the start and end frames:
        if f_start not in f_inds:
            f_inds.insert(0, f_start)

        if f_end not in f_inds:
            f_inds.append(f_end)

        if len(f_inds) == 2 and (f_end - f_start) > 1:
            # If we only have the start and end values, and they are more than
            # one frame apart, then we want at least to have the middle
            # frame as well.
            f_mid = int((f_start + f_end) / 2)
            f_inds.insert(0, f_mid)

        sec_inds = [(i / frate) for i in f_inds]

        out_idx = (fnl_tups.index(tup1)) + 1
        # Return the index + 1 corresponding to each tuple in the complete
        # tuple list. This represents the shot number.

        zf_out_idx = str(out_idx).zfill(4)
        # Z-filled string representing the clip number.

        # 'MovieTitleClip_0123'
        clip_dir = '{}clip_{}'.format(orig, zf_out_idx)
        os.makedirs(clip_dir, exist_ok=True)
        print('Clip number {}, {} through {}.'.format(zf_out_idx, f_start,
                                                      f_end))

        count = 0
        for idx2, sec_ind in enumerate(sec_inds):
            f_array = full_film.get_frame(sec_ind)
            img = Image.fromarray(f_array)
            wid, hgt = img.size
            if wid > 1080:
                img = img.resize((int(wid / 2), int(hgt / 2)))

            zf_idx_2 = str(f_inds[idx2]).zfill(6)
            # Z-filled string representing the actual frame number.

            # 'MovieTitle_000123.png'
            img_name = os.path.join(clip_dir, '{}_{}.png'.format(orig,
                                                                 zf_idx_2))
            img.save(img_name)
            if count % 10 == 0:
                print(img_name)
            count += 1


with mp.Pool(processes=mp.cpu_count()) as pool:
    pool.map(fnl_tup_processing, range(num_processes))
