import numpy as np
from tqdm import tqdm
from random import random
from tqdm import tnrange
import time
import pickle
import os

# def loadTextMatrix(fileName, numCols=16): # For Text document, it 16 byte Plaintext as an arrya
#     """
#     file format: hex values delimited by spaces
#     MAKE sure that there is no space at the end of line
#     """
#     return np.loadtxt(
#         fileName, dtype='uint8', delimiter=' ',
#         converters={_: lambda s: int(s, 16) for _ in range(numCols)})

path = 'Protected GIFT-COFB'

with open(path+'/fvrchoicefile.txt', 'r') as f:#change
    # groups = np.array([int(line.strip()) for line in f], dtype=np.uint8)
    content = f.read().strip()
    groups = np.array([int(char) for char in content], dtype=np.uint8)


SetPTArr = content

print(SetPTArr)


SetTraceArr =np.load(path+'/powerTraces.npy')
raw_traces=np.empty((2000,25000))


def adjustSampleSize(sampleLength, dataArray):
    # print "\tAdjusting Sample Size to ->" + str(sampleLength)
    temp = dataArray.shape
    newDataArray = dataArray
    arrLen = temp[0]
    # print "Array Length --> " + str(arrLen)
    if (arrLen == sampleLength):

        return dataArray

    elif (arrLen > sampleLength):

        diff = arrLen - sampleLength
        for count in range(0, diff):
            newDataArray = np.delete(newDataArray, -1, 0)
        return newDataArray

    elif (arrLen < sampleLength):

        diff = sampleLength - arrLen
        for count in range(0, diff):
            newDataArray = np.append(newDataArray, 0)
        return newDataArray

measurementFile = open(path+'/powerTraces.npy', 'r')
for traceCount in range (0,2000):
        #print "traceCount= " + str(traceCount)
        tempArrayMeasurement = np.load(path+'/powerTraces.npy')
        tempArrayMeasurement = adjustSampleSize(25000, tempArrayMeasurement)
        raw_traces[traceCount,:] = tempArrayMeasurement

print(raw_traces)
print(raw_traces.shape)


# Load your data
# traces = np.load("powerTraces.npy")#change

traces = raw_traces

# Load group data from the .txt file
with open(path+"/fvrchoicefile.txt", 'r') as f:#change
    # groups = np.array([int(line.strip()) for line in f], dtype=np.uint8)
    content = f.read().strip()
    groups = np.array([int(char) for char in content], dtype=np.uint8)

print(traces.shape[0])
print(groups.shape[0])
# Ensure the shapes match
assert traces.shape[0] == groups.shape[0], "Mismatch in number of traces and groups"

# Create a structured array
dtype = [('trace', 'float32', (25000,)), ('group', 'u1')]
combined_data = np.zeros(traces.shape[0], dtype=dtype)

# Populate the structured array
combined_data['trace'] = traces
combined_data['group'] = groups

# Save the combined data


file_path = os.path.join(path, 'Traces_1.dat')
with open(file_path, 'w') as f:
    f.write('')

combined_data.tofile(path+"/Traces_1.dat")



# Load the data from the .dat file
data = np.fromfile(path+"/Traces_1.dat", dtype=dtype)


# Print some sample data to inspect
print("First 5 traces and their group values:")
for i in range(5):
    print(f"Trace {i+1}: {data['trace'][i]}")
    print(f"Group {i+1}: {data['group'][i]}")


print(data['trace'].shape)
print(data['group'].shape)