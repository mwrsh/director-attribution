"""
This script is the third step of the workflow.  Given a directory of .txt
files, or a directory containing subdirectories, each of which are comprised of
.txt files, we do the following:
1. Output those same .txt files but with identifying
title/author information removed, leaving us only with compositional
strings.  Provided are different cutoffs for each composition--ranging
from 0 subjects (just shot distance) all the way to 5 subjects.
2. Provide the above, but culled to a number of settings: 1, 1/2, 1/3, 1/4,
and Full.  This corresponds to:
    * Removing any word that does not appear at least once in every text.
    * Removing any word that does not appear in at least 1/2 of texts.
    * Removing any word that does not appear in at least 1/3 of texts.
    * Removing any word that does not appear in at least 1/4 of texts.
    * Removing nothing.
3. Run Burrows', Cosine, Eder, and Quadratic delta analyses of each
possibility, and output .png of the dendrogram of the resulting cluster
analyses.  Each dendrogram includes useful data on the analyses' accuracy.

The 'filepath' argument must point toward either a directory containing only
.txt files, or a directory containing only subdirectories, each of which
itself contains only .txt files.

Each .txt file should contain only compositional strings, each separated by
a newline character; that is, the .txt files output by the second step in our
workflow, face_composition_token.py.  These compositional strings are of the
format:
[identifying data]DistXXAnnBnnCnnDnnEnn
i.e.
MScorsese_KingComedyclip_1143DistXCAXIBXICXIDXIEXI
"""

import argparse
import delta
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
# Significant amount of warnings from matplotlib/delta that are not worth
# addressing.


ap = argparse.ArgumentParser()
ap.add_argument('-f', '--filepath', type=str, required=True,
                help='Path to the file directory.')
args = vars(ap.parse_args())


def subdirs(path):
    # Yield directory names not starting with '.' under given path.
    for entry in os.scandir(path):
        if not entry.name.startswith('.') and entry.is_dir():
            yield entry.name


def pseudo_token(text_file, limit):
    # Converts a .txt file of line-by-line filenames into a list of 'words',
    # which creates a list akin to the tokenize function in NLTK, but for our
    # cinematic purposes. We then remove the film identifiers, thus returning
    # just the compositional data string.

    # Function expects our lines to be of the form:
    # [identifying data]DistXXAnnBnnCnnDnnEnn
    lines = [line.rstrip('\n') for line in open(text_file)]
    diff = (5 - limit) * 3
    # We split the string at different spots depending on how many subects
    # will be analyzed compositionally.
    out = [line[line.find('Dist'):(line.find('Dist')) + (21 - diff):] for
           line in lines]
    return out


def pseudo_cull(dir_list, thresh):
    # PyDelta's culling function does not easily allow for analysis of a
    # culled corpus, so this is just an ersatz way to do it on our own.
    for sub_dir in dir_list:
        text_list = []  # Viable text files
        word_list = []  # Unique word list
        word_dict = {}  # The number of texts in which a word appears
        cut_list = []   # The words that will be removed

        # The pre-cull directory is at this point of the form ./{}limitFull,
        # and we want to change it to ./{}limit{}
        out_dir = os.path.join(os.path.dirname(sub_dir), '{}{}'.format(
                               os.path.basename(sub_dir), thresh))
        out_dir = out_dir.replace('limitFull', 'limit')
        os.makedirs(out_dir, exist_ok=True)

        for poss_text in os.listdir(sub_dir):
            if not poss_text.startswith('.') and poss_text.endswith('.txt'):
                text_list.append(os.path.join(sub_dir, poss_text))

        comp_thresh = (1 / thresh) * len(text_list)
        # This will find the number of texts we need a word to be in before it
        # escapes culling.  So, if we have 24 texts in our corpus and our
        # threshold is 4, we will be searching for 1/4th of 24.  Any word
        # that does not appear in at least 6 texts, then, won't make the cut.

        for text in text_list:
            lines = [line.rstrip('\n') for line in open(text)]
            # Get all of the words in each text.
            for line in lines:
                if line not in word_list:
                    # Determine the unique word list.
                    word_list.append(line)

        for word in word_list:
            word_dict[word] = 0
            # Now loop back through our unique word list.
            for text in text_list:
                # Re-open each text ...
                lines = [line.rstrip('\n') for line in open(text)]
                if word in lines:
                    word_dict[word] += 1
                    # ... and increment for each word if encountered.

        for key in word_dict:
            if word_dict[key] < comp_thresh:
                cut_list.append(key)
            # So now we have a list of which words we will be removing.

        for text in text_list:
            culled = []  # Each text gets its own final list.
            lines = [line.rstrip('\n') for line in open(text)]
            for line in lines:
                if line not in cut_list:
                    culled.append(line)

            cull_title = os.path.basename(text)
            cull_title_full = os.path.join(out_dir, cull_title)

            with open(cull_title_full, 'w') as cull_out_txt:
                for winner in culled:
                    cull_out_txt.write(str(winner))
                    cull_out_txt.write('\n')


if len(list(subdirs(args['filepath']))) == 0:
    # i.e. if we are just looking at a single folder, rather than a folder
    # of folders.
    main_folder = args['filepath']

    for i in range(6):
        snip = os.path.join(main_folder, '{}limitFull'.format(i))
        os.makedirs(snip, exist_ok=True)

    cwd_list = os.listdir(main_folder)
    work_list = []
    for file in cwd_list:
        if not file.startswith('.') and file.endswith('.txt'):
            work_list.append(os.path.join(main_folder, file))

    for i in range(6):
        # Iterate over possible substring limits for the compositional cuts
        # from pseudo_token(), above.
        for file in work_list:
            title = os.path.basename(file)
            snip = os.path.join(main_folder, '{}limitFull'.format(i))
            title_full = os.path.join(snip, title)
            shots = pseudo_token(file, i)
            with open(title_full, 'w') as out_txt:
                for shot in shots:
                    out_txt.write(str(shot))
                    out_txt.write('\n')

    cwd_list = os.listdir(main_folder)

    cull_list = [1, 2, 3, 4, 'Full']
    # This cull list allows us to determine where to limit the overall
    # vocabulary of the dataset.  If the cull is set to 2, for example, then we
    # remove all words that do not appear in at least 1/2 of texts in the set.
    # If the cull is 'Full,' then we make no changes.

    snip_list = [
        (os.path.join(main_folder, '0limitFull')),
        (os.path.join(main_folder, '1limitFull')),
        (os.path.join(main_folder, '2limitFull')),
        (os.path.join(main_folder, '3limitFull')),
        (os.path.join(main_folder, '4limitFull')),
        (os.path.join(main_folder, '5limitFull'))
    ]
    pseudo_cull(snip_list, 1)
    pseudo_cull(snip_list, 2)
    pseudo_cull(snip_list, 3)
    pseudo_cull(snip_list, 4)
    # Apologies for the sloppy work above, but unexplained errors were
    # occurring when iterating rather than just doing this manually.

    out_path_full = os.path.join(main_folder, 'output_all')
    out_path_best = os.path.join(main_folder, 'output_best')
    os.makedirs(out_path_full, exist_ok=True)
    os.makedirs(out_path_best, exist_ok=True)

    deltas = ['burrows', 'cosine', 'eder', 'quad']

    count = 1 * 6 * 5 * 4
    # Count for tracking where we are at in the analysis process:
    # Number of testing subfolders * number of limits * number of cull
    # possibilities * number of deltas.

    comp_count = 0

    for delt in deltas:
        out_path = os.path.join(main_folder, 'output_{}'.format(delt))
        os.makedirs(out_path, exist_ok=True)

        for cull_set in cull_list:
            culled_dirs = [
                (os.path.join(main_folder, '0limit{}'.format(cull_set))),
                (os.path.join(main_folder, '1limit{}'.format(cull_set))),
                (os.path.join(main_folder, '2limit{}'.format(cull_set))),
                (os.path.join(main_folder, '3limit{}'.format(cull_set))),
                (os.path.join(main_folder, '4limit{}'.format(cull_set))),
                (os.path.join(main_folder, '5limit{}'.format(cull_set)))
            ]

            process_list = []
            for idx, snip in enumerate(culled_dirs):
                raw_corp = delta.Corpus(snip)
                most_freq_w = raw_corp.get_mfw_table(raw_corp.shape[1])

                if delt == 'burrows':
                    distances = delta.functions.burrows(most_freq_w)
                elif delt == 'cosine':
                    distances = delta.functions.cosine_delta(most_freq_w)
                elif delt == 'eder':
                    distances = delta.functions.eder(most_freq_w)
                else:
                    distances = delta.functions.quadratic(most_freq_w)
                cluster = delta.Clustering(distances)
                fcluster = cluster.fclustering()
                fclust_eval = fcluster.evaluate()
                process_list.append((cluster, fclust_eval))
                # A tuple comprised of the cluster and its own data.

            v_meas_list = []
            for tup in process_list:
                v_meas = tup[1].loc['V Measure']
                v_meas_list.append(v_meas)

            v_array = np.array(v_meas_list)
            quant = np.percentile(v_array, 67)
            # We will color the textbox based on whether the V Measure is
            # above the 67th quantile for that set.

            for idx, ob in enumerate(process_list):
                comp_count += 1
                print('Graphing: {} of {}'.format(comp_count, count))
                plt.figure(figsize=(15, 7))
                name = '{}_{}limit_{}cull.png'.format(delt, idx, cull_set)
                full_name = os.path.join(out_path, name)
                coll_name = os.path.join(main_folder,
                                         'output_all/{}'.format(name))
                best_name = os.path.join(main_folder,
                                         'output_best/{}'.format(name))
                delta.Dendrogram(ob[0])
                fig = plt.gcf()
                ax = (plt.gcf().get_axes())[0]
                # Grab the axes so that we can properly place the textbox.
                v_meas = ob[1].loc['V Measure']
                # The V Measure is the best measure of clustering accuracy,
                # as far as I know, and so we will color the textbox based
                # on whether it is a good or bad result.
                dfs = ob[1].to_string()
                if v_meas >= quant:
                    plt.text(0.05, 0.95, dfs, transform=ax.transAxes,
                             size=10,
                             ha='left', va='top',
                             bbox=dict(boxstyle='square',
                                       ec=(0.5, 1., 0.5),
                                       fc=(0.8, 1., 0.5),
                                       )
                             )
                    fig.savefig(full_name)
                    fig.savefig(coll_name)
                    fig.savefig(best_name)
                    plt.close()

                else:
                    # Different color textbox if the delta had a low V measure.
                    plt.text(0.05, 0.95, dfs, transform=ax.transAxes,
                             size=10,
                             ha='left', va='top',
                             bbox=dict(boxstyle='square',
                                       ec=(1., 0.5, 0.5),
                                       fc=(1., 0.8, 0.8),
                                       )
                             )

                    fig.savefig(full_name)
                    fig.savefig(coll_name)
                    plt.close()

else:
    main_folder = args['filepath']

    list_subs = list(subdirs(main_folder))
    # To generate our snipped compositional strings:
    for sub in list_subs:
        sub_full = os.path.join(main_folder, sub)

        for i in range(6):
            snip = os.path.join(sub_full, '{}limitFull'.format(i))
            os.makedirs(snip, exist_ok=True)

        cwd_list = os.listdir(sub_full)
        work_list = []
        for file in cwd_list:
            if not file.startswith('.') and file.endswith('.txt'):
                work_list.append(os.path.join(sub_full, file))

        for i in range(6):
            # Iterate over possible substring limits for the compositional cuts
            # from pseudo_token(), above.
            for file in work_list:
                title = os.path.basename(file)
                snip = os.path.join(sub_full, '{}limitFull'.format(i))
                title_full = os.path.join(snip, title)
                shots = pseudo_token(file, i)
                with open(title_full, 'w') as out_txt:
                    for shot in shots:
                        out_txt.write(str(shot))
                        out_txt.write('\n')

    # Refresh our subdirectory list, as we've added quite a bit of new material
    # since last time.
    list_subs = list(subdirs(main_folder))

    cull_list = [1, 2, 3, 4, 'Full']
    # This cull list allows us to determine where to limit the overall
    # vocabulary of the dataset.  If the cull is set to 2, for instance, then
    # we remove all words that do not appear in at least 1/2 of texts in the
    # set. If the cull is 'Full,' then we make no changes.

    for sub in list_subs:
        sub_full = os.path.join(main_folder, sub)
        snip_list = [
            (os.path.join(sub_full, '0limitFull')),
            (os.path.join(sub_full, '1limitFull')),
            (os.path.join(sub_full, '2limitFull')),
            (os.path.join(sub_full, '3limitFull')),
            (os.path.join(sub_full, '4limitFull')),
            (os.path.join(sub_full, '5limitFull'))
            ]
        pseudo_cull(snip_list, 1)
        pseudo_cull(snip_list, 2)
        pseudo_cull(snip_list, 3)
        pseudo_cull(snip_list, 4)

    deltas = ['burrows', 'cosine', 'eder', 'quad']

    count = len(list_subs) * 6 * 5 * 4
    # Count for tracking where we are at in the analysis process:
    # Number of testing subfolders * number of limits * number of cull
    # possibilities * number of deltas.

    for sub in list_subs:
        sub_full = os.path.join(main_folder, sub)
        out_path_full = os.path.join(sub_full, 'output_all')
        out_path_best = os.path.join(sub_full, 'output_best')
        os.makedirs(out_path_full, exist_ok=True)
        os.makedirs(out_path_best, exist_ok=True)

    comp_count = 0

    for delt in deltas:
        for sub in list_subs:
            sub_full = os.path.join(main_folder, sub)
            out_path = os.path.join(sub_full, 'output_{}'.format(delt))

            os.makedirs(out_path, exist_ok=True)

            for cull_set in cull_list:
                culled_dirs = [
                    (os.path.join(sub_full, '0limit{}'.format(cull_set))),
                    (os.path.join(sub_full, '1limit{}'.format(cull_set))),
                    (os.path.join(sub_full, '2limit{}'.format(cull_set))),
                    (os.path.join(sub_full, '3limit{}'.format(cull_set))),
                    (os.path.join(sub_full, '4limit{}'.format(cull_set))),
                    (os.path.join(sub_full, '5limit{}'.format(cull_set)))
                ]

                process_list = []
                for idx, snip in enumerate(culled_dirs):
                    raw_corp = delta.Corpus(snip)
                    most_freq_w = raw_corp.get_mfw_table(raw_corp.shape[1])

                    if delt == 'burrows':
                        distances = delta.functions.burrows(most_freq_w)
                    elif delt == 'cosine':
                        distances = delta.functions.cosine_delta(most_freq_w)
                    elif delt == 'eder':
                        distances = delta.functions.eder(most_freq_w)
                    else:
                        distances = delta.functions.quadratic(most_freq_w)
                    cluster = delta.Clustering(distances)
                    fcluster = cluster.fclustering()
                    fclust_eval = fcluster.evaluate()
                    process_list.append((cluster, fclust_eval))
                    # A tuple comprised of the cluster and its own data.

                v_meas_list = []
                for tup in process_list:
                    v_meas = tup[1].loc['V Measure']
                    v_meas_list.append(v_meas)

                v_array = np.array(v_meas_list)
                quant = np.percentile(v_array, 67)
                # We will color the textbox based on whether the V Measure is
                # above the 67th quantile for that set.

                for idx, ob in enumerate(process_list):
                    comp_count += 1
                    print('Graphing: {} of {}'.format(comp_count, count))
                    plt.figure(figsize=(15, 7))
                    name = '{}_{}limit_{}cull.png'.format(delt, idx, cull_set)
                    full_name = os.path.join(out_path, name)
                    coll_name = os.path.join(sub_full,
                                             'output_all/{}'.format(name))
                    best_name = os.path.join(sub_full,
                                             'output_best/{}'.format(name))
                    delta.Dendrogram(ob[0])
                    fig = plt.gcf()
                    ax = (plt.gcf().get_axes())[0]
                    v_meas = ob[1].loc['V Measure']
                    dfs = ob[1].to_string()
                    if v_meas >= quant:
                        plt.text(0.05, 0.95, dfs, transform=ax.transAxes,
                                 size=10,
                                 ha='left', va='top',
                                 bbox=dict(boxstyle='square',
                                           ec=(0.5, 1., 0.5),
                                           fc=(0.8, 1., 0.5),
                                           )
                                 )
                        fig.savefig(full_name)
                        fig.savefig(coll_name)
                        fig.savefig(best_name)
                        plt.close()

                    else:
                        # Different color textbox if the delta had a low
                        # V measure.
                        plt.text(0.05, 0.95, dfs, transform=ax.transAxes,
                                 size=10,
                                 ha='left', va='top',
                                 bbox=dict(boxstyle='square',
                                           ec=(1., 0.5, 0.5),
                                           fc=(1., 0.8, 0.8),
                                           )
                                 )
                        fig.savefig(full_name)
                        fig.savefig(coll_name)
                        plt.close()
