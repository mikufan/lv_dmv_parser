import subprocess
from multiprocessing import Pool
import os
import time
import numpy as np
import sys

train_langs = 'et-la_ittb-no-fi-grc-nl-en-de-la-bg-it-hi-fr-eu-sl'
train_langs = train_langs.split('-')

command = 'python src/eval_from_loaded_model.py --loaded_model_idx 4  --child_neural --em_type em --cvalency 2 --do_eval --ml_comb_type 4 ' +\
'--stc_model_type 1 --em_iter 1  --non_neural_iter 20 --non_dscrm_iter 60 --epochs 0 --function_mask ' +\
'--train et-la_ittb-no-fi-grc-nl-en-de-la-bg-it-hi-fr-eu-sl --load_model --dev '

print(command)

for i in train_langs:
    os.system(command + i)
