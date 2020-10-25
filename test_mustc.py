import json
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import sys
import os

from dataloaders import ImageCaptionDataset, ImageCaptionDatasetHDF5
from evaluate_segments import  parse_file, compute_metrics
from run_unit_analysis import (load_dataset, get_word_ali, get_code_ali,
                               prepare_data, STOP_WORDS, comp_code_to_wordprec,
                               comp_word_to_coderecall)
from run_utils import load_audio_model_and_state
from steps.plot import (load_raw_spectrogram, plot_spec_and_alis,
                        plot_precision_recall, plot_num_words_above_f1_threshold)
from steps.unit_analysis import (print_code_to_word_prec, print_word_by_code_recall,
                                 comp_code_word_f1, print_code_stats_by_f1, 
                                 print_word_stats_by_f1, count_high_f1_words,
                                 compute_topk_avg_f1)

def parse_ali(list_segments):
    all_segments = []
    for segment in list_segments.data:
        all_segments.append(float(segment[1]))
    return all_segments

ref_segments = parse_file("/disk/scratch1/ramons/image_seg/data/mustc/mustc.fr.seg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


models = [       
                ("RDVQ_01000_01100_01110","{2}->{2,3}->{2,3,4}"),
                ("RDVQ_00000_00100","{}->{3}"), 
                ("RDVQ_00000_01000","{}->{2}"),
                ("RDVQ_00000_01100","{}->{2,3}"),
                ("RDVQ_00100","{3}"),
                ("RDVQ_00100_01100","{3}->{2,3}"),
                ("RDVQ_01000","{2}"),
                ("RDVQ_01000_01100","{2}->{2,3}"),
                ("RDVQ_01100","{2,3}")]


# load data
hdf5_path = '/disk/scratch1/ramons/image_seg/ResDAVEnet-VQ/data/mustc/enfr/mustc_dev.json'


#dataset = ImageCaptionDataset(hdf5_path)
dataset = ImageCaptionDatasetHDF5(hdf5_path)

window = 0.03

with open("./mustc_dev_word_"+str(int(window*1000))+"ms","w") as output_results:
    output_results.write("model,layer,precision,recall,f1,os\n")
    for model in models:

        exp_dir = '/disk/scratch1/ramons/image_seg/ResDAVEnet-VQ/exps/models/'+model[0]
        audio_model = load_audio_model_and_state(exp_dir=exp_dir)
        audio_model = audio_model.to(device)
        audio_model = audio_model.eval()

        print('Successfully loaded dataset and model at %s' % time.asctime())

        layer_ids = model[1].split("->")[-1].replace("{","").replace("}","")
        if(layer_ids == ""):
            continue
        else:
            for layer_id in layer_ids.split(","):

                hyp_segments = {}
                for sample_idx in range(len(dataset)):
                    code_alip, wav_id = get_code_ali(audio_model, 'quant'+layer_id, dataset, sample_idx, device)
                    code_ali =  code_alip.get_sparse_ali()
                    utt_id = str(wav_id).replace("b'","").replace(".wav'","")
                    hyp_segments[utt_id] = parse_ali(code_ali)


                precision, recall, f1, os = compute_metrics(hyp_segments, ref_segments, window)

                output_results.write(model[1].replace(",",".")+","+layer_id+","+str(precision)+","+str(recall)+","+str(f1)+","+str(os)+"\n")

