import wfdb
import numpy as np
import pandas as pd
import ast
import scipy.signal as signal
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


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
        



        def aggregate_diagnostic(y_dic):
            tmp = []
            
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))

        

        # load and convert annotation data
        Y = pd.read_csv('ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        signals = load_raw_data(Y)
        # lowering the precion
        # signals = signals.astype('float32')

        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv('scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        """ it applies the aggregate_diagnostic function to each value in the 'scp_codes' column """
        Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)


        return signals, Y






    def splitdata(self, X, Y):


        # Split data into train and test
        test_fold = 10
        # Train
        X_train = X[np.where(Y.strat_fold != test_fold)]
        y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
        # Test
        X_test = X[np.where(Y.strat_fold == test_fold)]
        y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

#         print(agg_df)

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




    def preprocessData(self, X_train, y_train, X_test):
        # Preprocess the data
        # Reshape the training signals into a 2D array
        train_signals = X_train.reshape((X_train.shape[0], -1))

        # Reshape the test signals into a 2D array
        test_signals = X_test.reshape((X_test.shape[0], -1))

        # Handle missing values in the training signals
        imputer = SimpleImputer(strategy='mean')
        train_signals = imputer.fit_transform(train_signals)

        # Handle missing values in the test signals
        test_signals = imputer.transform(test_signals)

        # Standardize the features
        scaler = StandardScaler()
        train_signals = scaler.fit_transform(train_signals)
        test_signals = scaler.transform(test_signals)

        return train_signals, y_train, test_signals
    



    def classifyData(self, X_train, y_train, X_test, classifier):

        # Convert the target labels to a list
        y_train_list = [label[0] if len(label) > 0 else 'None' for label in y_train]

        # Convert the target labels to a binary array
        mlb = MultiLabelBinarizer()
        y_train_bin = mlb.fit_transform(y_train_list)

        # Preprocess the data
        train_signals, _, test_signals = self.preprocessData(X_train, y_train_bin, X_test)

        # Flatten the target variable
        y_train_flat = y_train_bin.flatten()

        # Train the classifier
        classifier.fit(train_signals, y_train_flat)

        # Predict on the test data
        y_pred = classifier.predict(test_signals)

        return y_pred