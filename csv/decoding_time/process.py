import numpy as np
import os
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f))]


import seaborn as sns
sns.set(font_scale=1.5)  # Make sure everything is readable.
sns.set_style("whitegrid")

all_dicts = {}
temp, gen_ll, beam = {}, {}, {}
mapper = {'temp':temp, 'gen_ll':gen_ll, 'beam':beam}
title = {'temp':'Temperature', 'gen_ll':'Generator Rejection', 'beam':'Beam Search'}
color = {'temp':'red', 'gen_ll':'green', 'beam':'blue'}

for f in onlyfiles:
    if '.csv' in f:
        dd = {}
        lines = open(f, 'rb').read().splitlines()
        keys, valuess = lines[0], lines[1:]
        keys = keys.split(',')
        for key in keys:
            dd[key] = []

        for values in valuess:
            try:values = map(float, values.split(','))
            except:
                import pdb; pdb.set_trace()
                x =1 
            for key, value in zip(keys, values):
                dd[key] += [value]

        all_dicts[f] = dd


for filename, dd in all_dicts.items():
    time = True if 'time' in filename else False
    if not time: 
        assert 'score' in filename, filename
    if time:
        id = filename.split('eval')[-1].split('time')[0][1:-1]
        mapper[id]['time'] = dd
    else:
        id = filename.split('eval')[-1].split('lm_score')[0][1:-1]
        mapper[id]['score'] = dd

final = {}
for dd in mapper:
    step = mapper[dd]['time']['Step']
    time = mapper[dd]['time']['Value']
    lm   = mapper[dd]['score']['Value']
    tt = list(zip(step, time, lm))
    tt.sort(key=lambda x : x[0])
    final[dd] = tt

import matplotlib.pyplot as plt

for method in ['temp', 'gen_ll', 'beam']:
    print(method)
    dd = final[method]
    start = 0 if 'll' in method else 0
    end = -6 if 'beam' in method else len(dd)
    ys = [x[1] for x in dd][start:end]
    xs = [y[2] for y in dd][start:end]
    mtd = plt.scatter if 'beam' in method else plt.plot
    mtd(xs, ys, label=title[method], color=color[method])

plt.ylabel('Decoding Time (s)')
plt.xlabel('LM score')
plt.legend(markerfirst=False, frameon=False)
plt.yscale('log')
plt.xlim(left=1.5)
#plt.show()
plt.savefig('decoding_time.pdf', bbox_inches='tight')
