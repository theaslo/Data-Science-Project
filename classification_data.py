import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from intervaltree import Interval,IntervalTree
from IPython.display import Audio
import matplotlib.pyplot as plt 
#import sounddevice as sd
import codecs, json
#def load_csv():
df=pd.read_csv('musicnet_metadata.csv')

def get_composer():
    fs = 44100 
    scores_array=np.load(open('musicnet.npz','rb'),encoding='latin1',allow_pickle=True)     
    #text_file = open("sample.txt", "w")
    test_keys=['1788','1789','2659']#, '1789','2659', '2127', '1818', '1819'
    new_data={}
    composers = []
    note_list=[]
    for aKey in scores_array.keys():
    #for aKey in test_keys:
        lk=df.loc[df.id==int(aKey), 'composer'].values[0]
        composers.append(lk)
        hatter = []

        X,Y = scores_array[aKey]
        ylength=len(Y)

        for t in range(ylength):#ylength
            alength= len(Y[fs*t])
            notes_array=[]
            for i in range(alength):#
                (start,end,(instrument,note,measure,beat,note_value)) = sorted(Y[fs*t])[i]
                notes_array.append((str(instrument), str(note), str(measure), note_value))
            #We want to sort by instrument 
            #because the positional embedding might have trouble otherwise 
            notes_array=sorted(notes_array, key = lambda x: x[0]) 
            return_array=[]
            # Now that we are sorted by instrument, remove tuple structure
            for i in notes_array:
                return_array.append(' '.join(i))
            listToStr = ', '.join([str(elem) for elem in return_array])+'. '

            hatter.append(listToStr)
        #print(hatter)
        #concat all seconds into one time piece
        listToStr2 = ''.join([str(elem) for elem in hatter])
        #now append time piece to a list that is indexed as the composer
        note_list.append(listToStr2)
    #make dataframe with composer and time piece
    new_data = {'Composer':composers, 'piece':note_list}
    new_df=pd.DataFrame(new_data)

    
    new_df.head()
    new_df.to_csv("data_4_classification.csv", index=False)
            
if __name__ == "__main__":
    get_composer()
    #load_csv()
    #get_note() 