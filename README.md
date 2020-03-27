# director-attribution
Using facial detection as a means toward cinematic authorship attribution.

## Modules Used
argparse  
dvt (https://github.com/distant-viewing/dvt)  
datetime  
math  
matplotlib  
multiprocessing  
numpy  
opencv-contrib-python (https://www.pyimagesearch.com/2018/09/19/pip-install-opencv/)  
os  
pillow  
pydelta (https://github.com/cophi-wue/pydelta)  
statistics  
string  
warnings  

## Goals:
1. To use facial recognition to detect the composition of film shots in a (nearly) automated manner.  
2. Output the compositions above in a human-readable, statistically analysable manner.  
3. Using delta statistical models (thus far Burrows', Eder, Cosine, and Quadratic), test if shot composition frequencies in a film serve as a thumbprint of directorial style.  

## Why?:
If one of these delta methods can consistently distinguish between directors, then it means that the vocabulary-based methods of authorship attribution used for literary works can be directly transferred to cinematic style, serving as a quantitativ ebasis for auteur theory and quantitative proof that the shot is indeed the lexical unit of cinema.  
  
Moreover, an analysis such as this would show that directorial style is based in sub-intentional decisions that are less easily imitable, inasmuch as it will largely be based on the ratio of directors' most-frequently used, most-mundane compositions. This is analogous to literary stylometric methods that successfully distinguish authorship based on the frequency distribution of function words (and, if, a, the, an, etc.). While any clumsy modern director can evoke Scorsese or Kubrick's most-iconic shots, the ratio of their "function shots," as it were, would betray true authorship and, perhaps, true influence.

## Caveats:
This is a work in progress. Initial results suggest that it indeed distinguishes authors successfully, but my dataset remains limited and, for now, is only based on color films, as I am as-yet unfamiliar with how to detect cuts in black and white film.  

These scripts have been written to be capable of running *relatively* quickly on my own 2018 MacBook Air. I am generally able to process an entire film and have it ready for delta analysis wwithin 30 to 90 minutes, with only 5 minutes of that requiring my direct attention. As noted in my comments within each script, there are certainly places where we could enhance our precision and thoroughness at the expense of speed, and I would encourage users with access to a dedicated computer vision rig to experiment with my suggestions and let me know their results!

## How?:
**_Acut_detector.py_** offers to the user 5 60-second clips taken from 1/6th, 2/6th, 3/6th, etc. through their chosen video file. The user then manually counts the number of shots in the clip, and the script uses these manual counts to determine the ideal parameters to properly detect cuts for the entire film. Outputs a subdirectory for each shot, each of which contains one frame per 0.5 seconds (minimum 1 frame).  

**_Bface_detect_comp.py_** processes the output from the above, determining shot distance and composition based on the size and position of the most-prominent faces in the shot. Returns its results as a series of representative strings of the format:  
`[film title]_clip[number]Dist[XX]AnnBnnCnnDnnEnn'`  
where XX can be e**X**treme **C**lose-up, **C**lose-**U**p, **M**edium **C**lose-up, **M**edium **S**hot, or **M**edium **L**ong; and where Ann, Bnn, Cnn ... refer to the locations of the up to 5 most-prominent subjects in the shot. The 'nn' portion is of the structure X#, where # is a letter between A and I, representing the 9 sectors of a typical rule of thirds-divided frame.  
Or, less confusingly, if the 345th clip in Martin Scorsese's film *King of Comedy* had a medium shot distance, with its largest subject located at middle-center, its second-largest at mid-left, its third-largest at bottom-right, with no other detected subjects:  
`MartinScorsese_KingofComedy_clip345DistMSAXEBXDCXIDnnEnn'`  
Optionally, this script can also provide us illustrative images for each shot by averaging the pixel data and drawing our face detection bounding boxes over the resulting incredibly blurry and strange output image.
  
**_Cdelta_testing.py_** prepares a clean version of our composition strings, separate from the identifying film information. It then uses delta calculations to provide statistical analyses of our corpus based on a number of potential parameters, returns dendrograms of each result, and separates out the parameters that returned the best V Measure.

Optionally, **_Ddelta_single.py_** allows users to run a single delta calculation using a single set of parameters, rather than looping over all possible options as in the above.
