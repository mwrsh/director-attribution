"""
This script is the (optional) final step of the workflow.  Given a
directory of .txt files as would have been provided by Bface_detect_comp.py,
we provide the delta results based on the user's
selected number of subjects, cull point, and delta measure.  Testing for
optimum settings can be done using the script Cdelta_testing.py

The 'filepath' argument must point toward either a directory containing only
.txt files of the structure created by Bface_detect_comp.py, or a directory
containing only subdirectories, which themselves contain only .txt files of
that structure.
"""

import argparse
import delta
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
# Significant amount of warnings from matplotlib/delta that are not worth
# addressing.


ap = argparse.ArgumentParser()
ap.add_argument('-f', '--filepath', type=str,
                required=True,
                help='Path to the file directory.')
ap.add_argument('-s', '--subjects', choices=['0', '1', '2', '3', '4', '5'],
                required=True,
                help='Number of subjects to count in compositions.')
ap.add_argument('-c', '--cull', choices=['1', '2', '3', '4', 'Full'],
                required=True,
                help='Cull setting.')
ap.add_argument('-d', '--delta', choices=['burrows', 'cosine', 'eder', 'quad'],
                required=True,
                help='Choice of delta function.')
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


main_folder = args['filepath']
subj_set = int(args['subjects'])
if args['cull'] == 'Full':
    cull_set = 'Full'
else:
    cull_set = int(args['cull'])
delta_set = args['delta']

snip = os.path.join(main_folder, '{}limitFull'.format(subj_set))
os.makedirs(snip, exist_ok=True)

cwd_list = os.listdir(main_folder)
work_list = []
for file in cwd_list:
    if not file.startswith('.') and file.endswith('.txt'):
        work_list.append(os.path.join(main_folder, file))

# Iterate over possible substring limits for the compositional cuts
# from pseudo_token(), above.
for file in work_list:
    title = os.path.basename(file)
    title_full = os.path.join(snip, title)
    shots = pseudo_token(file, subj_set)
    with open(title_full, 'w') as out_txt:
        for shot in shots:
            out_txt.write(str(shot))
            out_txt.write('\n')

cwd_list = os.listdir(main_folder)

if cull_set != 'Full':
    pseudo_cull([snip], cull_set)
    # We only need to bother generating culled text files if we are not
    # using the 'Full' setting.

out_path_full = os.path.join(main_folder, 'output_final')
os.makedirs(out_path_full, exist_ok=True)

culled_dir = snip.replace('limitFull', 'limit{}'.format(cull_set))

raw_corp = delta.Corpus(culled_dir)
most_freq_w = raw_corp.get_mfw_table(raw_corp.shape[1])

if delta_set == 'burrows':
    distances = delta.functions.burrows(most_freq_w)
elif delta_set == 'cosine':
    distances = delta.functions.cosine_delta(most_freq_w)
elif delta_set == 'eder':
    distances = delta.functions.eder(most_freq_w)
else:
    distances = delta.functions.quadratic(most_freq_w)

cluster = delta.Clustering(distances)
fcluster = cluster.fclustering()
fclust_eval = fcluster.evaluate()
process_tup = (cluster, fclust_eval)

name = '{}_{}limit_{}cull.png'.format(delta_set, subj_set, cull_set)
final_name = os.path.join(out_path_full, name)

plt.figure(figsize=(15, 7))
delta.Dendrogram(process_tup[0])
fig = plt.gcf()
ax = (plt.gcf().get_axes())[0]
# Grab the axes so that we can properly place the textbox.
dfs = process_tup[1].to_string()
plt.text(0.05, 0.95, dfs, transform=ax.transAxes,
         size=10,
         ha='left', va='top',
         bbox=dict(boxstyle='square',
                   ec=(0.5, 1., 0.5),
                   fc=(0.8, 1., 0.5),
                   )
         )
fig.savefig(final_name)
plt.close()
