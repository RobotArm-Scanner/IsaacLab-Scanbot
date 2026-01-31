#!/usr/bin/env bash
wget -r -np -nH --cut-dirs=4 -R "index.html*" -c -P "$(dirname "$0")/datasets" https://cv8-archive.cvlab.kr/ext_sdc1/home/scanbot/datasets/episode_e2.t1_f1__siwon1/
