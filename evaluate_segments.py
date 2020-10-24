import sys
import math
from random import seed
from random import randint
# seed random number generator
seed(1)
# generate
import argparse


def parse_file(filepath):
    dict_file={}
    with open(filepath) as inputfile:
        for linefile in inputfile.readlines():
            splited_line = linefile.strip().split()
            utt_id = splited_line[0]
            dict_file[utt_id] = splited_line[1].split(",")
    return dict_file


def time_to_frame(timemark, framerate):
    # 1 second = 1000 milliseconds
    # kaldi uses windows of 10 millis
    return int(timemark*1000/10)

def count_correct_boundaries(ref,hyp,tolerance):
    number_correct = 0
    for re in ref:
        for hy in hyp:
            if abs(float(re) - float(hy)) <= tolerance:
                number_correct += 1
                break
    return number_correct




def compute_metrics(hyp_dict, ref_dict, tolerance):


    n_boundaries_ref = 0

    previous_id = ""
    first = True
    ref_boundaries={}


    n_boundaries_ref = 0
    n_boundaries_seg = 0
    n_boundaries_correct = 0
    number_utterances=0
    os_acum=0

    for hyp_key in hyp_dict.keys():
        if(hyp_key not in ref_dict):
            print(hyp_key+" NOT present!")
            continue

        utterance_id= hyp_key 
        ref_vector = ref_dict[utterance_id]
        hyp_vector = hyp_dict[utterance_id]
        n_boundaries_ref += len(ref_vector)

        number_utterances+=1
        
        n_boundaries_seg += len(hyp_vector)

        n_boundaries_correct += count_correct_boundaries(ref_vector, hyp_vector, tolerance)


    os_sent=(os_acum/number_utterances)*100

    if(n_boundaries_correct == 0):
        precision=0
        recall=0
    else:
        recall = (float(n_boundaries_correct)/n_boundaries_ref)*100
        precision = (float(n_boundaries_correct)/n_boundaries_seg)*100

    os=((float(n_boundaries_seg)/float(n_boundaries_ref))-1)*100
    ra=math.sqrt((100-recall)**2+(os**2))
    rb=(-os+recall-100)/math.sqrt(2)
    final_r=1-((abs(ra)+abs(rb))/200)

    if precision + recall != 0:
        f = 2*precision*recall / (precision + recall)
    else:
        f = 0

    return round(float(precision),2), round(float(recall),2), round(float(f),2), round(float(os),2)

