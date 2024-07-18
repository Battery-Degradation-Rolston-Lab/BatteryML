# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import torch
import numpy as np
from typing import List
from batteryml.builders import LABEL_ANNOTATORS
from batteryml.data.battery_data import BatteryData
from batteryml.data.battery_data import CycleData
from batteryml.data.battery_data import CyclingProtocol
from .base import BaseLabelAnnotator  # Ensure BaseLabelAnnotator is imported

@LABEL_ANNOTATORS.register()
class SOHLabelAnnotator(BaseLabelAnnotator):
    def __init__(self, voltage_threshold: float = 3.0, outlier_threshold: float = 2.5):
        self.voltage_threshold = voltage_threshold
        self.outlier_threshold = outlier_threshold

    def remove_outliers(self, data: np.ndarray, threshold: float) -> np.ndarray:
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / std)
        return data[z_scores <= threshold]

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        return (data - np.mean(data)) / np.std(data)

    def process_cell(self, cell_data: BatteryData) -> torch.Tensor:
        if not cell_data.cycle_data:
            raise ValueError("BatteryData must contain cycle data to calculate SOH.")

        # Extract additional features
        discharge_capacities = np.array([cycle.discharge_capacity_in_Ah[-1] for cycle in cell_data.cycle_data if cycle.discharge_capacity_in_Ah])
        initial_voltages = np.array([cycle.voltage_in_V[0] for cycle in cell_data.cycle_data if cycle.voltage_in_V])
        temperatures = np.array([cycle.temperature_in_C[0] for cycle in cell_data.cycle_data if cycle.temperature_in_C])

        # Ensure all arrays have the same length before removing outliers
        min_length = min(len(discharge_capacities), len(initial_voltages), len(temperatures))
        discharge_capacities = discharge_capacities[:min_length]
        initial_voltages = initial_voltages[:min_length]
        temperatures = temperatures[:min_length]

        # Remove outliers
        discharge_capacities = self.remove_outliers(discharge_capacities, self.outlier_threshold)
        initial_voltages = self.remove_outliers(initial_voltages, self.outlier_threshold)
        temperatures = self.remove_outliers(temperatures, self.outlier_threshold)

        # Ensure all arrays still have the same length after removing outliers
        min_length = min(len(discharge_capacities), len(initial_voltages), len(temperatures))
        discharge_capacities = discharge_capacities[:min_length]
        initial_voltages = initial_voltages[:min_length]
        temperatures = temperatures[:min_length]

        # Normalize features
        discharge_capacities = self.normalize_data(discharge_capacities)
        initial_voltages = self.normalize_data(initial_voltages)
        temperatures = self.normalize_data(temperatures)

        # Combine features for model input
        features = np.stack([discharge_capacities, initial_voltages, temperatures], axis=1)

        # Simple SOH calculation (consider using a more complex model here)
        max_discharge_capacity = np.max(discharge_capacities)
        nominal_capacity = cell_data.nominal_capacity_in_Ah
        soh_percentage = (max_discharge_capacity / nominal_capacity) * 100 if nominal_capacity > 0 else float('nan')

        label = torch.tensor(soh_percentage)
        return label
 