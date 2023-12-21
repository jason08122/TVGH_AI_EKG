from xml2npy_filter import decode_and_filter

import numpy as np
import os
import shutil
# from time import sleep
import json
import pandas as pd

from my_inference import ai_test
from lvsdinf.lvsdclassifier import LVSDClassifier

lvsd = LVSDClassifier(checkpoint='./lvsdinf/weights/checkpoint_0104.pth.tar')

def fixed_columns(dataframe):
    return dataframe[['id', 'date', 'sddd_model_pred', 'sddd_model_prob_sd', 'sddd_model_prob_dd', 'sddd_model_prob_others', 'sd_model_pred', 'sd_model_prob_sd', 'sd_model_prob_others']]

def ekg_inference(ecg, pid, ekg_date):
    label_csv_file = '/home/Diagnosis.csv'
    sddd_res, sddd_Diagnosis = ai_test(ecg)
    sddd_res = [int(num*100)/100.0 for num in sddd_res]

    ecg = np.expand_dims(ecg, axis=0)
    ecg = ecg.astype(np.float32)

    sd_res, sd_Diagnosis = lvsd.predict(ecg)
    sd_res = [int(num*100)/100.0 for num in sd_res]
    diagnosis = ['HRrEF', 'HFpEF', 'Normal']

    df = pd.read_csv(label_csv_file)
    new_row = {'id': pid, 'date': ekg_date, 'sddd_model_pred': diagnosis[sddd_Diagnosis], 'sddd_model_prob_sd': sddd_res[0], 
                'sddd_model_prob_dd': sddd_res[1], 'sddd_model_prob_others': sddd_res[2], 'sd_model_pred': sd_Diagnosis,
                'sd_model_prob_sd': sd_res[1], 'sd_model_prob_others': sd_res[0]
    }

    # Append the new row to the DataFrame
    df.loc[len(df)] = new_row
    df = fixed_columns(df)
    df = df.drop_duplicates(subset=['id', 'date'], keep='last')
    df.to_csv(label_csv_file)

def AddPatient2Json(pid, ekg_date):
    json_file = '/home/Patient.json'

    if not os.path.exists(json_file):
        data = {}
    else:
        with open(json_file, 'r') as file:
            data = json.load(file)

    # with open(json_file, 'r') as file:
    #     data = json.load(file)
    print(f'pid: {pid} date: {ekg_date}')
    pid_str = str(pid)

    if pid_str in data:
        data[pid_str].append(ekg_date)
    else:
        data[pid_str] = [ekg_date]

    tmp = list(set(data[pid_str]))

    sorted_dates = sorted(tmp, key=lambda x: tuple(map(int, x.split('-'))))

    data[pid_str] = sorted_dates

    with open(json_file, 'w') as file:
        json.dump(data, file, indent=2)  # 'indent=2' for pretty formatting (optional)

def move_xml_files(source_folder, destination_folder):
    # Check if the source and destination folders exist
    if not os.path.exists(source_folder) or not os.path.exists(destination_folder):
        print("Source or destination folder does not exist.")
        return

    # List all files in the source folder
    files = os.listdir(source_folder)
    cnt = len(files)

    # Iterate through the files and move XML files to the destination folder
    for file in files:
        if file.lower().endswith('.xml'):
            source_file = os.path.join(source_folder, file)
            destination_file = os.path.join(destination_folder, file)
            
            ecg, pid, ekg_date = decode_and_filter(source_file)
            # print(f'Patient ID: {pid} EKG date: {ekg_date}')
            
            # Update Patient.json file
            AddPatient2Json(pid, ekg_date)

            # Update Diagnosis.csv file
            ekg_inference(ecg, pid, ekg_date)          

            # Move XML file to the destination folder
            shutil.move(source_file, destination_file)
            print(f"Moved '{file}' to '{destination_folder}'")
    
    return cnt

def force_move_xml_files(): # new control api 
    source_folder = '/home/xml_buffer'
    destination_folder = '/home/xml_finished'
    # Check if the source and destination folders exist
    if not os.path.exists(source_folder) or not os.path.exists(destination_folder):
        print("Source or destination folder does not exist.")
        return

    # List all files in the source folder
    files = os.listdir(source_folder)
    cnt = len(files)

    # Iterate through the files and move XML files to the destination folder
    for file in files:
        if file.lower().endswith('.xml'):
            source_file = os.path.join(source_folder, file)
            destination_file = os.path.join(destination_folder, file)
            
            ecg, pid, ekg_date = decode_and_filter(source_file)
            # print(f'Patient ID: {pid} EKG date: {ekg_date}')
            
            # Update Patient.json file
            AddPatient2Json(pid, ekg_date)

            # Update Diagnosis.csv file
            ekg_inference(ecg, pid, ekg_date)          

            # Move XML file to the destination folder
            shutil.move(source_file, destination_file)
            print(f"Moved '{file}' to '{destination_folder}'")
    
    return cnt

if __name__ == '__main__':
    while (1):
        print('Start decode and inference')
        
        source_folder = '/home/xml_buffer'
        destination_folder = '/home/xml_finished'

        try:
            cnt = move_xml_files(source_folder, destination_folder)
            cnt = 0 if cnt == None else cnt

            print(f'{cnt} xml files decoded')
        except:
            print(f'Decoded error')
        
        # sleep(5)


