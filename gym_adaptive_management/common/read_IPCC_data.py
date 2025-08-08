from importlib.resources import files
import pandas as pd

def load_IPCC_dataset():
	# Assuming 'IPCC_data.csv' is in 'gym_adaptive_management/data/'
	data_path = files('gym_adaptive_management.data').joinpath('IPCC_data.csv')
	with data_path.open('r') as f:
	    # Process your data, e.g., using pandas
	    df = pd.read_csv(f)
	return df
