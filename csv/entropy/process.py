import os
import matplotlib.pyplot as plt

import numpy as np
import os
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(os.getcwd()) if isfile(join(os.getcwd(), f)) and 'csv' in f]

import seaborn as sns
sns.set(font_scale=1.5)  # Make sure everything is readable.
sns.set_style("whitegrid")

def get_values():
    nll_arr, ent_arr = None, None
    for f in onlyfiles:
        print(f)
        values = open(f, 'r').read().splitlines()
     
        if True: 
            nll = 'NLL' in f
            if nll:
                print(f)
                label = 'NLL Test'
            else:
                label = 'Entropy'

            # remove header
            values = values[1:]

            # remove discriminator pretraining values
            values = values[:10] + values[15:]

            values = np.array(map(float, values))    
           
            if nll:
                nll_arr = values
            else:
                ent_arr = values


    return nll_arr, ent_arr

nll_values, ent_values = get_values()
assert nll_values.shape == ent_values.shape
lin = np.arange(1, nll_values.shape[0]+1)

size=3
# make graph
plt.subplot(2, 1, 1)
plt.plot(lin, nll_values, color='blue', linewidth=size)
plt.axvline(x=10, color='red', linestyle='--')
plt.ylabel('NLL test')
plt.xlim(left=0.)
plt.xticks([0, 20, 40, 60], visible=False)

plt.subplot(2, 1, 2)
plt.plot(lin, ent_values, color='blue', linewidth=size)
plt.axvline(x=10, color='red', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Entropy')
plt.xlim(left=0)
plt.xticks([0, 20, 40, 60])
#plt.show()
plt.savefig('entropy.pdf', bbox_inches='tight')
