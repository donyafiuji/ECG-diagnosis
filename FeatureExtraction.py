import numpy as np
from scipy.signal import cwt, find_peaks
import matplotlib.pyplot as plt
import pywt as pw
import random
from scipy.stats import describe
from scipy.interpolate import interp1d
from scipy import signal





class QRS(): 




    def __init__(self):


        self.max_bpm = 200
        self.sampling_rate = 100 




    # wavelet transform, thresholding, and peak detection to extract R-peaks
    
    def R_peak_detection(self, original_signal):

        
        ecg_signal = original_signal
        ## Stationary Wavelet Transform
        coeffs = pw.swt(ecg_signal, wavelet = "haar", level=2, start_level=0, axis=-1)
        d2 = coeffs[1][1] ##2nd level detail coefficients


        ## Threhold the detail coefficients
        avg = np.mean(d2)
        std = np.std(d2)
        sig_thres = [abs(i) if abs(i)>2.0*std else 0 for i in d2-avg]

        ## Find the maximum modulus in each window
        window = int((60.0/self.max_bpm)*self.sampling_rate)
        sig_len = len(ecg_signal)
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
        merge_width = int((0.2)*self.sampling_rate)
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
        window_check = int(self.sampling_rate/6)
        #signal_normed = np.absolute((signal-np.mean(signal))/(max(signal)-min(signal)))
        r_peaks = [0]*len(qrs)

        for i,loc in enumerate(qrs):
            start = max(0,loc-window_check)
            end = min(sig_len,loc+window_check)
            wdw = np.absolute(ecg_signal[start:end] - np.mean(ecg_signal[start:end]))
            pk = np.argmax(wdw)
            r_peaks[i] = start+pk


#         time = np.arange(len(ecg_signal)) / self.sampling_rate    
#         plt.figure(figsize=(20,5))
#         plt.plot(time, ecg_signal, label='ECG Signal')
#         plt.scatter(time[r_peaks], ecg_signal[r_peaks], c='r', marker='o', label='R-peaks')
#         plt.xlabel('Time (s)')
#         plt.ylabel('Amplitude')
#         plt.title(f'ECG Signal {0} with Detected R-peaks')
#         plt.legend()
#         plt.grid(True)
#         plt.show()


        return r_peaks
    



    """ Computes the differences between consecutive R peak locations to obtain the RR intervals. """

    def RR_Intervals(self, Rpeaks):

        intervals = np.diff(Rpeaks) / self.sampling_rate

        return intervals
    



    """ just the lead II is considered for thr calculations: """

    def calculate_hrv_features(self, rr_intervals):


        """ basic HRV features """

        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
        nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50)
        pnn50 = nn50 / len(rr_intervals) * 100.0

        # Compute frequency domain HRV features using Welch's method
        freq, psd = signal.welch(rr_intervals, fs=1.0)

        # Compute power in different frequency bands
        vlf = np.trapz(psd[(freq >= 0.0033) & (freq < 0.04)])
        lf = np.trapz(psd[(freq >= 0.04) & (freq < 0.15)])
        hf = np.trapz(psd[(freq >= 0.15) & (freq <= 0.4)])

        # Compute normalized power in LF and HF bands
        total_power = vlf + lf + hf
        lf_norm = lf / total_power
        hf_norm = hf / total_power

        # Compute additional transformed features
        lf_hf_ratio = lf / hf
        lf_hf_ratio_log = np.log(lf_hf_ratio)
        total_power_log = np.log(total_power)
        lf_hf_ratio_log_norm = (lf_hf_ratio_log - np.mean(lf_hf_ratio_log)) / np.std(lf_hf_ratio_log)

        # Collect all HRV features
        hrv_features = [
            mean_rr,
            std_rr,
            rmssd, 
            nn50, 
            pnn50, 
            total_power, 
            vlf, 
            lf, 
            hf, 
            lf_norm,
            hf_norm, 
            lf_hf_ratio, 
            lf_hf_ratio_log, 
            total_power_log, 
            lf_hf_ratio_log_norm
        ]
        


        return hrv_features