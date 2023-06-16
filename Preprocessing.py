import wfdb
import numpy as np
import pandas as pd
import ast





class loadData():


    def __init__(self):

        self.sampling_rate=100
        self.path='/'


    def load_raw_data(self, df):

        if self.sampling_rate == 100:
            data = [wfdb.rdsamp(f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(f) for f in df.filename_hr]

        data = np.array([signal for signal, meta in data])
        meta_data = data[0][1]
        return data, meta_data
    

    def example_physionet(self):


        # load and convert annotation data
        Y = pd.read_csv('ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        # Original_ecg, meta_data = self.load_raw_data(Y)

        # # Load scp_statements.csv for diagnostic aggregation
        # agg_df = pd.read_csv('scp_statements.csv', index_col=0)
        # agg_df = agg_df[agg_df.diagnostic == 1]


        # Apply diagnostic superclass
        # Y['diagnostic_superclass'] = Y.scp_codes.apply(self.aggregate_diagnostic)

        # # Split data into train and test
        # test_fold = 10
        # # Train
        # X_train = Original_ecg[np.where(Y.strat_fold != test_fold)]
        # y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
        # # Test
        # X_test = X[np.where(Y.strat_fold == test_fold)]
        # y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

        return Y,Y.scp_codes

    
    def aggregate_diagnostic(self, y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in self.agg_df.index:
                tmp.append(self.agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    