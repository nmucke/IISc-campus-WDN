import pdb
import numpy as np
import pandas as pd
import networkx as nx
import wntr
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from ML_for_WDN.data_utils import clean_dataframes, load_data

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder



class CampusDataset(torch.utils.data.Dataset):
    """Dataset"""

    def __init__(
        self,
        dataframes: list,
        with_leak: bool = True,
        preprocessor = None,
    ):
        """
        Args:
            data_files (list): List of data files to be loaded.
        """
        self.with_leak = with_leak

        self.preprocessor = preprocessor

        self.dataframes = dataframes

        
        leak_pipe = {
            'leak_1': ('J-32', 'J-86'),#, 'P-49'),#'P-2',
            'leak_2': ('J-44', 'J-35'),#), 'P-2'),#'P-49',
            'leak_3': ('J-15', 'J-72'),#, 'P-26'),#'P-26',
        }

        if self.with_leak:

            self.leak_dataframe_ids = {}
            id_counter = 0
            for i, key in enumerate(['no_leak', 'leak_1', 'leak_2', 'leak_3']):
                self.leak_dataframe_ids[key] = np.arange(id_counter, id_counter + len(self.dataframes[i]))
                id_counter += len(self.dataframes[i])

            pars_list = []
            for i, df in enumerate(self.dataframes):
                pars_list.append(self._get_pars_for_one_df(df, leak_pipe=i))
            
            self.pars = self._combine_pars(pars_list)

            self.pars = torch.tensor(self.pars, dtype=torch.int32)

            self.dataframes = pd.concat(self.dataframes, axis=0, ignore_index=True)
        
        else:
            self.dataframes = self.dataframes[0]

    def _get_pars_for_one_df(self, df, leak_pipe: int):

        pars = torch.zeros((len(df), 1))

        pars[:, 0] = int(leak_pipe)

        return pars
    
    def _combine_pars(self, pars_list):
        return torch.cat(pars_list, dim=0)

    def __len__(self):
        return len(self.dataframes)
    
    def __getitem__(self, idx):
        
        sample = self.dataframes.iloc[idx, :]

        if self.preprocessor is not None:
            sample = self.preprocessor.transform(sample.values.reshape(1, -1))
            sample = sample.reshape(-1)

        if self.with_leak:
            pars = self.pars[idx, :]
            return torch.tensor(sample, dtype=torch.get_default_dtype()), pars

        else:
            return torch.tensor(sample, dtype=torch.get_default_dtype())

