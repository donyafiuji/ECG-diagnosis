import numpy as np
from scipy.signal import cwt, find_peaks
import matplotlib.pyplot as plt
import pywt as pw
import random






class QRS(): 




    def __init__(self):
        
        plt.figure(figsize=(10, 2))


    # wavelet transform, thresholding, and peak detection to extract R-peaks
    
    def peak_detection(self, sig):

        

        # Computation the wavelet coefficients using the Stationary Wavelet Transform
        coeffs = pw.swt(sig, wavelet = "db4", level=3, start_level=0, axis=-1)
        d3 = coeffs[2][1] ##2nd level detail coefficients

        sample_rate = 100
        max_bpm = 66 


        ## Threhold the detail coefficients
        avg = np.mean(d3)
        std = np.std(d3)
        sig_thres = [abs(i) if abs(i)>4.0*std else 0 for i in d3-avg]

        ## Find the maximum modulus in each window
        window = int((60.0/max_bpm)*sample_rate)
        sig_len = len(sig)
        n_windows = int(sig_len/window)
        modulus,qrs = [],[]

        ##Loop through windows and find max modulus
        for i in range(n_windows):
            start = i*window
            end = min([(i+1)*window,sig_len])
            mx = max(sig_thres[start:end])
            if mx>0:
                modulus.append( (start + np.argmax(sig_thres[start:end]),mx))


        ## Merge if within max bpm
        merge_width = int((0.2)*sample_rate)
        i=0
        while i < len(modulus)-1:
            ann = modulus[i][0]
            if modulus[i+1][0]-modulus[i][0] < merge_width:
                if modulus[i+1][1]>modulus[i][1]: # Take larger modulus
                    ann = modulus[i+1][0]
                i+=1
                    
            qrs.append(ann)
            i+=1 

        
        ## Pin point exact qrs peak
        window_check = int(sample_rate/6)
        #signal_normed = np.absolute((signal-np.mean(signal))/(max(signal)-min(signal)))
        r_peaks = []

        for loc in qrs:
            start = max(0,loc-window_check)
            end = min(sig_len,loc+window_check)
            wdw = np.absolute(sig[start:end] - np.mean(sig[start:end]))
            pk = np.argmax(wdw)

            if wdw[pk] >= 0.3:
                r_peaks.append(start + pk)


        plt.plot(sig, color='blue', label='ECG Signal')
        plt.plot(r_peaks, sig[r_peaks], 'ro', markersize=5, label='R-Peaks')
        plt.show()

        return r_peaks