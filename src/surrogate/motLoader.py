import os
import tqdm
import numpy as np
import pandas as pd
# Data loader for .mot files

def read_header(mot_file,header_line=10): 
    if not os.path.isfile(mot_file): 
        return []

    try:         
        with open(mot_file, 'r') as f:
            lines = f.readlines()
            headers = lines[header_line]
            headers = headers.split()
            return headers
    except Exception as e:
        print(f"Unable to load headers for file:{mot_file}. Error:{e}")
        return [] 

def storage_to_numpy(storage_file, excess_header_entries=0):
    """Returns the data from a storage file in a numpy format. Skips all lines
    up to and including the line that says 'endheader'.
    Parameters
    ----------
    storage_file : str
        Path to an OpenSim Storage (.sto) file.
    Returns
    -------
    data : np.ndarray (or numpy structure array or something?)
        Contains all columns from the storage file, indexable by column name.
    excess_header_entries : int, optional
        If the header row has more names in it than there are data columns.
        We'll ignore this many header row entries from the end of the header
        row. This argument allows for a hacky fix to an issue that arises from
        Static Optimization '.sto' outputs.
    Examples
    --------
    Columns from the storage file can be obtained as follows:
        >>> data = storage2numpy('<filename>')
        >>> data['ground_force_vy']
    """
    # What's the line number of the line containing 'endheader'?
    f = open(storage_file, 'r')

    header_line = False
    for i, line in enumerate(f):
        if header_line:
            column_names = line.split()
            break
        if line.count('endheader') != 0:
            line_number_of_line_containing_endheader = i + 1
            header_line = True
    f.close()
    # With this information, go get the data.
    if excess_header_entries == 0:
        names = True
        skip_header = line_number_of_line_containing_endheader
    else:
        names = column_names[:-excess_header_entries]
        skip_header = line_number_of_line_containing_endheader + 1
    data = np.genfromtxt(storage_file, names=names,
            skip_header=skip_header)

    new_data = []
    for d in data:
        new_data.append(list(d))
    new_data = np.array(new_data)

    return data


def storage_to_dataframe(storage_file, headers):
    # Extract data
    data = storage_to_numpy(storage_file)
    data = np.array(data)
    new_data = []
    for d in data:
        new_data.append(list(d))
    new_data = np.array(new_data)
    header_mapping = {header:i for i,header in enumerate(headers)}

    out = pd.DataFrame(data=data['time'], columns=['time'])
    for count, header in enumerate(headers):
        out.insert(count + 1, header, new_data[:,count+1])    
    
    return out