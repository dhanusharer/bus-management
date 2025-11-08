import pandas as pd
import numpy as np

buses = [f'bus{i}' for i in range(1, 6)]
busstops = [f'busstop_{i}' for i in range(1, 10)]

data_rows = []
for bus in buses:
    for stop in busstops:
        onboard = np.random.randint(5, 20)
        deboarded = np.random.randint(1, onboard + 1)  # deboarded <= onboard
        data_rows.append([bus, stop, onboard, deboarded])

df = pd.DataFrame(data_rows, columns=['bus', 'busstop', 'onboard', 'deboarded'])
df.to_csv('bus_onboard_deboard_mock_data.csv', index=False)

print("Mock data CSV file 'bus_onboard_deboard_mock_data.csv' created successfully.")

