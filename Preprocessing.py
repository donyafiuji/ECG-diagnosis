import wfdb
import numpy as np
import pandas as pd
import ast
import scipy.signal as signal
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet


""" - Nyquist frequency is the maximum frequency that can be accurately represented
 in a digital signal sampled at a given rate """



class preprocess():



    def __init__(self):

        self.sampling_rate=100
        self.path='/'
        



  
    """ prepare data and split them to test and train """

    def loadData(self):


        def load_raw_data(df):
            if self.sampling_rate == 100:
                data = [wfdb.rdsamp(f) for f in df.filename_lr]
            else:
                data = [wfdb.rdsamp(f) for f in df.filename_hr]
            data = np.array([signal for signal, meta in data])
            return data
        

        # load and convert annotation data
        Y = pd.read_csv('ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data(Y)

        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv('scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        def aggregate_diagnostic(y_dic):
            tmp = []
            
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))

        # Apply diagnostic superclass
        Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

        # Split data into train and test
        test_fold = 10
        # Train
        X_train = X[np.where(Y.strat_fold != test_fold)]
        y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
        # Test
        X_test = X[np.where(Y.strat_fold == test_fold)]
        y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

        return X_train, y_train, X_test, y_test





    """  baseline wander removal = low-frequency drift """

    def highpassfilter(self, origin_signal ,sampling_rate , filter_order, cutoff_freq):

        nyquist_freq = 0.5 * sampling_rate
        cutoff_freq = cutoff_freq 
        filter_order = filter_order

        # Design a Butterworth highpass filter
        b, a = signal.butter(filter_order, cutoff_freq / nyquist_freq, btype='highpass')


        # Apply the filter to the ECG signal
        baseline_removed_ecg = signal.filtfilt(b, a, origin_signal)

        
        return baseline_removed_ecg 
    





    def bandpassfilter(self , origin_signal, sampling_rate ,filter_order ,lowcut_freq , highcut_freq): 


        highcut = highcut_freq
        lowcut = lowcut_freq 
        filter_order = filter_order  
        nyquist_freq = 0.5 * sampling_rate
        b, a = signal.butter(filter_order, [lowcut / nyquist_freq, highcut / nyquist_freq], btype='band')

        bandpassfiltered = signal.filtfilt(b, a, origin_signal)
   
        # plt.plot(origin_signal, label= 'original signal')
        # plt.plot(bandpassfiltered, label = 'bandpass filtered')
        # plt.show()

        return bandpassfiltered
    



    def movingaveragefilter(self ,original_signal):

        MovingAvarage_filtered = np.convolve(original_signal, np.ones((3,))/3, mode="same")

        # plt.plot(original_signal, label= 'original signal')
        # plt.plot(MovingAvarage_filtered, label = 'bandpass filtered')
        # plt.show()

        return MovingAvarage_filtered



    

    def wavelet_denoising(self, original_signal):

        denoised = denoise_wavelet(original_signal, method='BayesShrink', mode='soft', wavelet_levels=3, wavelet='sym8', rescale_sigma='True')
        
        return denoised
    



    def noise_representation(self, firstsignal, secondsignal):

        noise = firstsignal - secondsignal
        plt.figure(figsize=(20,5))
        plt.plot(firstsignal, color='b', label='Original Signal')
        plt.plot(noise, color='red', label='Noise')
        plt.show()