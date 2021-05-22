import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from intervaltree import Interval,IntervalTree
from IPython.display import Audio
import matplotlib.pyplot as plt 
import codecs, json


fs = 44100  

def make_dict():
    print("making dictionary")
    scores_array=np.load(open('musicnet.npz','rb'),encoding='latin1',allow_pickle=True)
    print("array loaded")

    #scores_dict=dict(scores_array)
    # scores_dict= dict(np.load(open('musicnet.npz','rb'),encoding='latin1',allow_pickle=True))
    print("dumping dictionary")
    #json.dump( scores_dict, open( "my_dict.json", 'w' ) )
    b = scores_array.tolist() # nested lists with same data, indices
    file_path = "my_dict.json" ## your path variable
    json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    print("Finished making dictionary")

def load_dict():
    pass

def split_dict():
    json.load( open( "file_name.json" ) )

if __name__ == "__main__":
     make_dict() 
