from tqdm import tqdm
from typing import List
from pathlib import Path
import pandas as pd
import numpy as np
from batteryml import BatteryData, CycleData, CyclingProtocol
from batteryml.builders import PREPROCESSORS
from batteryml.preprocess.base import BasePreprocessor

@PREPROCESSORS.register()
class CustomPreprocessor(BasePreprocessor):
    def process(self, parent_dir: str) -> List[BatteryData]:
        path = Path(parent_dir)
        excel_files = list(path.glob('*.xlsx'))

        batteries = []
        for file in tqdm(excel_files, desc='Processing Excel files'):
            # Read data from Excel file, skipping first 99 rows and remove empty rows
            data_df = pd.read_excel(file, skiprows=98).dropna(how='all')

            cycle_data_dict = {}
            for _, row in data_df.iterrows():
                cycle_number = row['cycle number']

                # Initialize dictionary for this cycle number if not already initialized
                if cycle_number not in cycle_data_dict:
                    cycle_data_dict[cycle_number] = {
                        'cycle_number': int(cycle_number),
                        'voltage_in_V': [],
                        'current_in_A': [],
                        'discharge_capacity_in_Ah': [],
                        'charge_capacity_in_Ah': [],
                        'time_in_s': [],
                        'temperature_in_C': [],
                        'internal_resistance_in_ohm': [],
                        'Qdlin': []
                    }

                # Append values to lists for each cycle number
                cycle_data_dict[cycle_number]['voltage_in_V'].append(float(row['Ecell/V']) if pd.notna(row['Ecell/V']) else np.nan)
                cycle_data_dict[cycle_number]['current_in_A'].append(float(row['control/mA']) if pd.notna(row['control/mA']) else np.nan)
                cycle_data_dict[cycle_number]['discharge_capacity_in_Ah'].append(float(row['Q discharge/mA.h']) if pd.notna(row['Q discharge/mA.h']) else np.nan)
                cycle_data_dict[cycle_number]['charge_capacity_in_Ah'].append(float(row['Q charge/mA.h']) if pd.notna(row['Q charge/mA.h']) else np.nan)
                cycle_data_dict[cycle_number]['time_in_s'].append(float(row['time/s']) if pd.notna(row['time/s']) else np.nan)
                cycle_data_dict[cycle_number]['temperature_in_C'].append(int(row['x']) if pd.notna(row['x']) else np.nan)
                cycle_data_dict[cycle_number]['internal_resistance_in_ohm'].append(float(row['R/Ohm']) if pd.notna(row['R/Ohm']) else np.nan)
                cycle_data_dict[cycle_number]['Qdlin'].append(float(row['Q charge/mA.h'] - row['Q discharge/mA.h']) if pd.notna(row['Q charge/mA.h']) and pd.notna(row['Q discharge/mA.h']) else np.nan)

            cycle_data = []
            for cycle_number, data in cycle_data_dict.items():
                avg_internal_resistance = np.nanmean(data['internal_resistance_in_ohm']) if data['internal_resistance_in_ohm'] else np.nan

                cycle_data.append(CycleData(
                    cycle_number=data['cycle_number'],
                    voltage_in_V=data['voltage_in_V'],
                    current_in_A=data['current_in_A'],
                    discharge_capacity_in_Ah=data['discharge_capacity_in_Ah'],
                    charge_capacity_in_Ah=data['charge_capacity_in_Ah'],
                    time_in_s=data['time_in_s'],
                    temperature_in_C=data['temperature_in_C'],
                    internal_resistance_in_ohm=avg_internal_resistance,
                    Qdlin=data['Qdlin']
                ))

            charge_protocol = [CyclingProtocol(rate_in_C=2.0, start_soc=0.0, end_soc=1.0)]
            discharge_protocol = [CyclingProtocol(rate_in_C=1.0, start_soc=1.0, end_soc=0.0)]

            form_factor = 'cylindrical_18650'
            anode_material = 'graphite'
            cathode_material = 'LCO'
            nominal_capacity_in_Ah = 2500  # Replace with actual nominal capacity
            min_voltage_limit_in_V = 2.7  # Replace with actual min voltage limit
            max_voltage_limit_in_V = 4.2  # Replace with actual max voltage limit

            batteries.append(BatteryData(
                cell_id=file.stem,
                cycle_data=cycle_data,
                charge_protocol=charge_protocol,
                discharge_protocol=discharge_protocol,
                form_factor=form_factor,
                anode_material=anode_material,
                cathode_material=cathode_material,
                nominal_capacity_in_Ah=nominal_capacity_in_Ah,
                min_voltage_limit_in_V=min_voltage_limit_in_V,
                max_voltage_limit_in_V=max_voltage_limit_in_V
            ))

        self.dump(batteries)
        self.summary(batteries)
        return batteries

    def __call__(self, parent_dir: str):
        batteries = self.process(parent_dir)
        self.dump(batteries)
        if not self.silent:
            self.summary(batteries)

    def dump(self, batteries: List[BatteryData]):
        if not self.silent:
            batteries = tqdm(
                batteries,
                desc=f'Dumping batteries to {str(self.output_dir)}')
        for battery in batteries:
            battery.dump(self.output_dir / f'{battery.cell_id}.pkl')

    def summary(self, batteries: List[BatteryData]):
        print(f'Successfully processed {len(batteries)} batteries.')

