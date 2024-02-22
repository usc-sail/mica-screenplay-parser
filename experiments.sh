#!/bin/bash

mode=$1

# mode=evaluate_simple
python run.py --mode=$mode # rule-based parser
python run.py --mode=$mode --trx # transformer-based parser

# mode=evaluate_issue
# rule-based parser
python run.py --mode=$mode --results_dir=results/evaluation_issue/rule --error=none --issue_file=noerror
python run.py --mode=$mode --results_dir=results/evaluation_issue/rule --issue_file=error
# transformer-based parser
# [TODO]

# mode=create_data
python run.py --mode=$mode

# mode=create_seq
python run.py --mode=$mode --seqlen=10

# mode=create_feats
python run.py --mode=$mode