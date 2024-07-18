# # Licensed under the MIT License.
# # Copyright (c) Microsoft Corporation.

import torch

from batteryml.builders import LABEL_ANNOTATORS
from batteryml.data.battery_data import BatteryData

from .base import BaseLabelAnnotator


# @LABEL_ANNOTATORS.register()
# class RULLabelAnnotator(BaseLabelAnnotator):
#     def __init__(self,
#                  eol_soh: float = 0.8,
#                  pad_eol: bool = True,
#                  min_rul_limit: float = 100.0):
#         self.eol_soh = eol_soh
#         self.pad_eol = pad_eol
#         self.min_rul_limit = min_rul_limit

#     def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
#         label, found_eol = 1, False
#         for cycle in cell_data.cycle_data:
#             label += 1
#             Qd = max(cycle.discharge_capacity_in_Ah)
#             if Qd <= cell_data.nominal_capacity_in_Ah * self.eol_soh:
#                 found_eol = True
#                 break

#         if not found_eol:
#             label = label + 1 if self.pad_eol else float('nan')

#         if label <= self.min_rul_limit:
#             label = float('nan')

#         label = torch.tensor(label)
#         print(label)
#         return label

# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

# import torch

# from batteryml.builders import LABEL_ANNOTATORS
# from batteryml.data.battery_data import BatteryData

# from .base import BaseLabelAnnotator


# @LABEL_ANNOTATORS.register()
# class RULLabelAnnotator(BaseLabelAnnotator):
#     def __init__(self,
#                  eol_soh: float = 0.8,
#                  pad_eol: bool = True,
#                  min_rul_limit: float = 100.0,
#                  decay_rate: float = 0.01):
#         """
#         Custom RUL Label Annotator with linear degradation model.

#         :param eol_soh: End of life state of health threshold (e.g., 0.8 for 80%)
#         :param pad_eol: Boolean indicating whether to pad end of life if not found
#         :param min_rul_limit: Minimum RUL limit below which RUL is set to NaN
#         :param decay_rate: Decay rate for capacity degradation
#         """
#         self.eol_soh = eol_soh
#         self.pad_eol = pad_eol
#         self.min_rul_limit = min_rul_limit
#         self.decay_rate = decay_rate

#     def calculate_rul(self, current_cycle: int, total_cycles: int) -> float:
#         """
#         Calculate RUL based on the linear degradation model.

#         :param current_cycle: The current cycle number
#         :param total_cycles: The total number of cycles
#         :return: Remaining useful life (RUL)
#         """
#         print(max(total_cycles - current_cycle, 0))
#         return max(total_cycles - current_cycle, 0)

#     def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
#         """
#         Process cell data to generate RUL labels.

#         :param cell_data: BatteryData object containing cycle data
#         :return: Tensor containing the RUL label
#         """
#         total_cycles = len(cell_data.cycle_data)
#         label = float('nan')

#         for cycle_num, cycle in enumerate(cell_data.cycle_data, start=1):
#             Qd = max(cycle.discharge_capacity_in_Ah)
#             if Qd <= cell_data.nominal_capacity_in_Ah * self.eol_soh:
#                 label = self.calculate_rul(cycle_num, total_cycles)
#                 break

#         if label == float('nan') and self.pad_eol:
#             label = self.calculate_rul(total_cycles + 1, total_cycles)

#         if label <= self.min_rul_limit:
#             label = float('nan')

#         label = torch.tensor(label)
#         print(label)
#         return label

# import torch
# from batteryml.annotators import BaseLabelAnnotator, LABEL_ANNOTATORS
# from batteryml.data import BatteryData

# import torch
# from batteryml.annotators import BaseLabelAnnotator, LABEL_ANNOTATORS
# from batteryml.data import BatteryData

@LABEL_ANNOTATORS.register()
class RULLabelAnnotator(BaseLabelAnnotator):
    def __init__(self,
                 eol_soh: float = 0.8,
                 pad_eol: bool = True,
                 min_rul_limit: float = 40):
        self.eol_soh = eol_soh
        self.pad_eol = pad_eol
        self.min_rul_limit = min_rul_limit
        print(f"Initialized with eol_soh={self.eol_soh}, pad_eol={self.pad_eol}, min_rul_limit={self.min_rul_limit}")

    def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
        label, found_eol = 1, False
        print(f"Initial label={label}, found_eol={found_eol}")

        for i, cycle in enumerate(cell_data.cycle_data):
            label += 1
            Qd = max(cycle.discharge_capacity_in_Ah)
            print(f"Cycle {i}: label={label}, Qd={Qd}")

            if Qd <= cell_data.nominal_capacity_in_Ah * self.eol_soh:
                found_eol = True
                print(f"EOL found at cycle {i}: Qd={Qd}, nominal_capacity={cell_data.nominal_capacity_in_Ah}, eol_soh={self.eol_soh}")
                break

        print(f"After cycle loop: label={label}, found_eol={found_eol}")

        if not found_eol:
            if self.pad_eol:
                label += 1
                print(f"Pad EOL applied: new label={label}")
            else:
                label = float('nan')
                print(f"Pad EOL not applied: label set to NaN")

        if label <= self.min_rul_limit:
            label = float('nan')
            print(f"Label below min RUL limit: label set to NaN, min_rul_limit={self.min_rul_limit}")

        label = torch.tensor(label)
        print(f"Final label: {label}")
        return label
