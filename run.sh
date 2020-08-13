#!/bin/bash

FILE=(item1 item2 item3)
FILE[0]="BlackBox_IDS_SDN.py"
FILE[1]="IDS_WGAN_SDN.py"
FILE[2]="generate_attacktraffic.py"

for f in ${FILE[@]}; do
    eval "python3 ${f}"
done