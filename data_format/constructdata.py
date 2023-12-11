import numpy as np

# Load your data

path = '../Traces/Protected GIFT-COFB/'
traces = np.load(path + "powerTraces.npy")#change


# Load group data from the .txt file
with open(path+"fvrchoicefile.txt", 'r') as f:#change
    # groups = np.array([int(line.strip()) for line in f], dtype=np.uint8)
    content = f.read().strip()
    groups = np.array([int(char) for char in content], dtype=np.uint8)
print(traces.shape)
print(groups.shape)
# Ensure the shapes match
assert traces.shape[0] == groups.shape[0], "Mismatch in number of traces and groups"

# Create a structured array
dtype = [('trace', 'float32', (25000,)), ('group', 'u1')]
combined_data = np.zeros(traces.shape[0], dtype=dtype)

# Populate the structured array
combined_data['trace'] = traces
combined_data['group'] = groups

# Save the combined data
combined_data.tofile(path + "Traces_1.dat")

# Load the data from the .dat file
data = np.fromfile(path + "Traces_1.dat", dtype=dtype)

# Print some sample data to inspect
print("First 5 traces and their group values:")
for i in range(5):
    print(f"Trace {i+1}: {data['trace'][i]}")
    print(f"Group {i+1}: {data['group'][i]}")

print(data['trace'].shape)
print(data['group'].shape)