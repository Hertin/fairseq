#!/bin/bash

source $(dirname ${CONDA_EXE})/../etc/profile.d/conda.sh
conda activate wav2bart # change it to your conda environment

stage=0        # start from 0 if you need to start from data preparation
stop_stage=100

. utils/parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

librilight_path="/mnt/Eextension/SST/dataset/librispeech_finetuning" # change the path to librilight data
manifest_path="$(pwd)/manifest/librilight10h" # path to save manifest

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Preparing Librilight Dataset"
    splits="train dev_other test"
    
    # use librilight 1h for testing
    python wav2vec_manifest.py "${librilight_path}/1h" --dest ${manifest_path} --ext flac --valid-percent 0
    mv "${manifest_path}/train.tsv" "${manifest_path}/test.tsv"
    rm "${manifest_path}/valid.tsv" # remove empty valid manifest
    
    # use librilight 9h for training and validation
    python wav2vec_manifest.py "${librilight_path}/9h" --dest ${manifest_path} --ext flac --valid-percent 0.1
    mv "${manifest_path}/valid.tsv" "${manifest_path}/dev_other.tsv" # rename the valid set
    
    # get label for librilight
    for split in ${splits}; do
        python libri_labels.py ${manifest_path}/${split}.tsv --output-dir manifest/librilight10h/ --output-name ${split}
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Training Fairseq Model"
    config_dir="$(pwd)/config/training"
    config_name="base_10h"
    w2v_path="$(pwd)/models/wav2vec_small.pt"
    bart_path="$(pwd)/models/bart.base"
    fairseq-hydra-train task.data=${manifest_path} task.bart_path=${bart_path} model.w2v_path=${w2v_path} model.bart_path=${bart_path} --config-dir ${config_dir} --config-name ${config_name}

fi
