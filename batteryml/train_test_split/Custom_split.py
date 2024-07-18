from typing import List, Tuple
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from batteryml.builders import TRAIN_TEST_SPLITTERS
from batteryml.train_test_split.base import BaseTrainTestSplitter
    
@TRAIN_TEST_SPLITTERS.register()
class CustomTrainTestSplitter(BaseTrainTestSplitter):
    def __init__(self, cell_data_path: str):
        BaseTrainTestSplitter.__init__(self, cell_data_path)
        test_ids = [
            'Channel A2', 'Channel A4', 'Channel A6', 'Channel A8'
            ]


        self.train_cells, self.test_cells = [], []

        for filename in self._file_list:
            if filename.stem in test_ids:
                self.test_cells.append(filename)
            else: 
                self.train_cells.append(filename)

    def split(self) -> Tuple[List, List]:
        return self.train_cells, self.test_cells
