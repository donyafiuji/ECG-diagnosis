from scipy.fft import fft, ifft 
import numpy as np
import matplotlib.pyplot as plt



""" 
Fourier transform is the most convenient tool when signal frequencies do not change in time.
if the frequencies that make up the signal vary over time, the most performant technique is a wavelet transform. 

"""


class Transformers(): 

    def __init__(self):
        
        plt.figure(figsize=(20, 5))


    def FFT(self,signal):


        sig_spectrum = np.fft.fft(signal)
        magnitude = np.abs(sig_spectrum)
        phase = np.angle(sig_spectrum)
        freq = np.fft.fftfreq(len(signal), d=0.01)
        plt.plot(freq, sig_spectrum)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('FFT')
        plt.show()
        


    # def DWT(self,signal):
        

