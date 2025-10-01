import mne
import numpy as np
import os
import h5py
from tqdm import tqdm


def load_data(sampling_rate, participant_id, chunk_size=10000):
    processed_path = f"data/{participant_id}/{participant_id}_processed_data_{sampling_rate}Hz.npy"
    if not os.path.exists(processed_path):
        raw = mne.io.read_raw_edf(
            f"data/{participant_id}/{participant_id}.edf",
            preload=False,
            verbose="INFO",
        )

        non_eeg_channels = ["ChEMG1", "ChEMG2", "RLEG-", "RLEG+", "LLEG-", "LLEG+", "EOG2", "EOG1", "ECG1", "ECG2"]
        raw.drop_channels(non_eeg_channels)

        raw.resample(sampling_rate)
        raw.set_eeg_reference("average")

        n_samples = raw.n_times
        n_channels = raw.info["nchan"]
        mm = np.lib.format.open_memmap(
            processed_path, mode="w+", dtype=np.float64, shape=(n_channels, n_samples)
        )
        write_pos = 0
        for start in tqdm(range(0, n_samples, chunk_size)):
            stop = min(start + chunk_size, n_samples)
            chunk = raw.get_data(start=start, stop=stop)
            mm[:, write_pos:write_pos + (stop - start)] = chunk
            write_pos += (stop - start)
        del mm 
        data = np.load(processed_path, mmap_mode="r")
    else:
        data = np.load(processed_path, mmap_mode="r")

    mat_path = (f"data/Artifact_Detection_Matrices/{participant_id}_artndxn.mat")

    with h5py.File(mat_path, "r") as f:
        artifact_detection_matrix = f["artndxn"][:]
    data_before_reshaping = data.shape[1]
    data = data[:, : artifact_detection_matrix.shape[0] * 30 * sampling_rate] #This is executed for the majority 
    data_after_reshaping = data.shape[1]
    artifact_detection_matrix = artifact_detection_matrix.T

    adm_before_reshaping = artifact_detection_matrix.shape[1]
    artifact_detection_matrix = artifact_detection_matrix[:,:(data.shape[1] // (30*sampling_rate))] #This is only executed for P5
    adm_after_reshaping = artifact_detection_matrix.shape[1]

    if data_before_reshaping != data_after_reshaping:
        print(f"Data reshaped from {data_before_reshaping} to {data_after_reshaping}.")
    if adm_before_reshaping != adm_after_reshaping:
        print(f"Artifact detection matrix reshaped from {adm_before_reshaping} to {adm_after_reshaping}.") 

    return data, artifact_detection_matrix

def load_cached_data(sampling_rate):
    participants = list(range(1, 30))
    data_list = []
    adm_list = []

    for i in participants:
        participant_id = f"EPCTL0{i}" if i < 10 else f"EPCTL{i}"
        dt, adm = load_data(sampling_rate, participant_id)  
        data_list.append(dt)
        adm_list.append(adm.astype(np.int8, copy=False))

    return data_list, adm_list