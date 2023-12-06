from sierraecg.lib import read_file
import numpy as np
import neurokit2 as nk
import xml.etree.ElementTree as ET
import os

def baselineFilter(data):
    outputs = []
    for leads in data:
        leads = nk.signal_filter(signal=leads, sampling_rate=500, lowcut=0.5, highcut=None, method="butterworth", order=5)
        leads = nk.signal_filter(signal=leads, sampling_rate=500, method="powerline")
        outputs.append(leads)

    return np.vstack(outputs)

def decode_and_filter(xml_file: str = 'test.xml') -> np.array:
    ecg = []
    npy_folder = '/home/npy'

    try:
        os.mkdir(npy_folder)
    except FileExistsError:
        pass

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except:
        return None, None, None

    patient_id = root[5][0][0].text
    patient_id = int(patient_id)
    echo_date = root[3].get('date')
    str_date = echo_date.replace('-', '')

    year = echo_date.split('-')[0]

    year_folder = f'{npy_folder}/{year}'

    try:
        os.mkdir(year_folder)
    except FileExistsError:
        pass

    save_name = f'{year_folder}/{str_date}_{str(patient_id)}.npy'

    f = read_file(xml_file)

    for lead in f.leads:
        ecg.append([ele/200.0 for ele in lead.samples])

    data = baselineFilter(ecg)
    ecg = np.array(ecg, dtype=np.float32)
    ecg = data[:, :5000]
    data = data[:, :5000]

    np.save(save_name, ecg)

    return data, patient_id, echo_date


if __name__ == '__main__':
    decode_and_filter()