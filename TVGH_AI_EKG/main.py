from fastapi import FastAPI, File, UploadFile
from my_inference import ai_test
from xml2npy_filter import decode_and_filter
from lvsdinf.lvsdclassifier import LVSDClassifier
import numpy as np
import pandas as pd
from Initializer import create_csv, create_folder, create_json
from fastapi.middleware.cors import CORSMiddleware
from XmlChecker import force_move_xml_files

import os
import json

def fixed_columns(dataframe):
    return dataframe[['id', 'date', 'diagnosis']]

def clear_directory(directory):
    # List all files and folders in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Check if it's a file and delete it
        if os.path.isfile(filepath):
            os.remove(filepath)
            print(f"Deleted file: {filepath}")
        
        # Check if it's a directory and delete it recursively
        elif os.path.isdir(filepath):
            clear_directory(filepath)  # Recursively clear subdirectory
            os.rmdir(filepath)  # Remove the empty directory
            print(f"Deleted directory: {filepath}")

lvsd = LVSDClassifier(checkpoint='./lvsdinf/weights/checkpoint_0104.pth.tar')

app = FastAPI(
    title='EKG AI Diagnosis',
    description='Using xml to diagnos sd dd'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

@app.post("/twoDiagnos/upload", tags=["twoDiagnos"])
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open('new.xml', 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    ecg, pid, ekg_date = decode_and_filter('new.xml')
    
    sddd_res, sddd_Diagnosis = ai_test(ecg)
    sddd_res = [int(num*100)/100.0 for num in sddd_res]

    ecg = np.expand_dims(ecg, axis=0)
    ecg = ecg.astype(np.float32)

    sd_res, sd_Diagnosis = lvsd.predict(ecg)
    sd_res = [int(num*100)/100.0 for num in sd_res]
    diagnosis = ['HFrEF', 'HFpEF', 'Normal']

    result = {
            "Patient_ID": pid, 
            "EKG_date": ekg_date, 
            "sddd_Diagno": diagnosis[sddd_Diagnosis],
            "sddd_Diagnosis": 
            {
                "HFrEF": sddd_res[0], 
                "HFpEF": sddd_res[1], 
                "Normal": sddd_res[2]
            }, 
            "sd_Diagno": sd_Diagnosis,
            "sd_Diagnosis":
            {
                "HFrEF": sd_res[1], 
                "Normal": sd_res[0]
            }
        }
    
    return result

@app.get("/twoDiagnos/patient/model/preds", tags=["twoDiagnos"]) # 指定 api 路徑 (get方法)
def GetDiagnosByPid(pid: str = '-1'):
    if pid == '-1':
        return {"message": "Data missing"}
    
    diagnosis_csv = '/home/Diagnosis.csv'
    patient_json = '/home/Patient.json'

    if not os.path.exists(patient_json):
        data = {}
    else:
        with open(patient_json, 'r') as file:
            data = json.load(file)
    
    if pid not in data:
        return {"message": "Patient ID not in the system"}
    
    ekg_dates = data[pid]

    df = pd.read_csv(diagnosis_csv)

    pid_int = int(pid)

    ret = {}

    for date in ekg_dates:
        res = df.loc[(df['id'] == pid_int) & (df['date'] == date)]
        
        sddd_res = res['sddd_model_pred'].values[0], res['sddd_model_prob_sd'].values[0], res['sddd_model_prob_dd'].values[0], res['sddd_model_prob_others'].values[0]
        sd_res = res['sd_model_pred'].values[0], res['sd_model_prob_sd'].values[0], res['sd_model_prob_others'].values[0]
        
        date_info = {
            'sddd_model_pred': sddd_res[0],
            'sddd_model_prob_sd': sddd_res[1],
            'sddd_model_prob_dd': sddd_res[2],
            'sddd_model_prob_others': sddd_res[3],
            'sd_model_pred': sd_res[0],
            'sd_model_prob_sd': sd_res[1],
            'sd_model_prob_others': sd_res[2]
        }

        ret[date] = date_info

    return ret

@app.post("/oneDiagnos/upload", tags=["oneDiagnos"])
def upload_2(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open('new.xml', 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    ecg, pid, ekg_date = decode_and_filter('new.xml')
    
    sddd_res, sddd_Diagnosis = ai_test(ecg)
    sddd_res = [int(num*100)/100.0 for num in sddd_res]

    ecg = np.expand_dims(ecg, axis=0)
    ecg = ecg.astype(np.float32)

    sd_res, sd_Diagnosis = lvsd.predict(ecg)
    sd_res = [int(num*100)/100.0 for num in sd_res]

    sd_prob = sd_res[1]
    dd_prob = 0.0 if sddd_res[1] == 0.0 else sd_res[0] * sddd_res[1] / (sddd_res[1]+sddd_res[2])
    normal_prob = 0.0 if sddd_res[2] == 0.0 else sd_res[0] * sddd_res[2] / (sddd_res[1]+sddd_res[2])
    dd_prob = int(dd_prob*100)/100.0  
    normal_prob = int(normal_prob*100)/100.0  

    if sd_prob > dd_prob and sd_prob > normal_prob:
        model_pred = 'HFrEF'
    elif dd_prob > sd_prob and dd_prob > normal_prob:
        model_pred = 'HFpEF'
    else:
        model_pred = 'Normal'

    result = {
            "Patient_ID": pid, 
            "EKG_date": ekg_date, 
            'model_pred': model_pred,
            'prob_sd': sd_prob,
            'prob_dd': dd_prob,
            'prob_normal': normal_prob,
        }
    
    return result

@app.get("/oneDiagnos/patient/model/preds", tags=["oneDiagnos"]) # todo
def GetOneDiagnosByPid(pid: str = '-1'):
    if pid == '-1':
        return {"message": "Data missing"}
    
    diagnosis_csv = '/home/Diagnosis.csv'
    patient_json = '/home/Patient.json'

    if not os.path.exists(patient_json):
        data = {}
    else:
        with open(patient_json, 'r') as file:
            data = json.load(file)
    
    if pid not in data:
        return {"message": "Patient ID not in the system"}
    
    ekg_dates = data[pid]

    df = pd.read_csv(diagnosis_csv)

    pid_int = int(pid)

    ret = {}

    for date in ekg_dates:
        res = df.loc[(df['id'] == pid_int) & (df['date'] == date)]
        
        sddd_res = res['sddd_model_pred'].values[0], res['sddd_model_prob_sd'].values[0], res['sddd_model_prob_dd'].values[0], res['sddd_model_prob_others'].values[0]
        sd_res = res['sd_model_pred'].values[0], res['sd_model_prob_sd'].values[0], res['sd_model_prob_others'].values[0]
        
        sd_prob = sd_res[1]
        if sddd_res[2] == 0.0 and sddd_res[3] == 0.0:
            dd_prob = sd_res[2] / 2
            normal_prob = sd_res[2] / 2
        else:
            dd_prob = 0.0 if sddd_res[2] == 0.0 else sd_res[2] * sddd_res[2] / (sddd_res[2]+sddd_res[3])
            normal_prob = 0.0 if sddd_res[3] == 0.0 else sd_res[2] * sddd_res[3] / (sddd_res[2]+sddd_res[3])
        
        dd_prob = int(dd_prob*100)/100.0  
        normal_prob = int(normal_prob*100)/100.0        
        
        if sd_prob > dd_prob and sd_prob > normal_prob:
            model_pred = 'HFrEF'
        elif dd_prob > sd_prob and dd_prob > normal_prob:
            model_pred = 'HFpEF'
        else:
            model_pred = 'Normal'

        # ['HFrEF', 'HFpEF', 'Normal']

        date_info = {
            'model_pred': model_pred,
            'prob_sd': sd_prob,
            'prob_dd': dd_prob,
            'prob_normal': normal_prob,
        }

        ret[date] = date_info

    return ret

@app.get("/twoDiagnos/doctor/label", tags=["twoDiagnos"]) # 指定 api 路徑 (get方法)
def submitLabel(id: int = -1, date: str = '-1', diagnosis: str = '-1'):
    if id == -1 or date == '-1' or diagnosis == '-1':
        return {"message": "Data missing"}

    if not os.path.exists('/home/label.csv'):
        create_csv()
    df = pd.read_csv('/home/label.csv')
    new_row = {'id': id, 'date': date, 'diagnosis': diagnosis}

    # Append the new row to the DataFrame
    df.loc[len(df)] = new_row
    df = fixed_columns(df)
    df = df.drop_duplicates(subset=['id', 'date'], keep='last')
    df.to_csv('/home/label.csv')
    
    return {"message": "label saved", "Patient_ID": id, "EKG_date": date, "Diagnosis": diagnosis}


@app.get("/control/initial", tags=["control"]) # 指定 api 路徑 (get方法)
def initial_folder_file():

    try:
        create_csv()
    except:
        return {"message": "Create csv error"}
    try:
        create_folder()
    except:
        return {"message": "Create folder error"}    
    try:
        create_json()
    except:
        return {"message": "Create json error"}
    try:
        clear_directory('/home/npy')
    except:
        return {"message": "Clear npy folder error"}
    # try:
    #     clear_directory('/home/xml_buffer')
    # except:
    #     return {"message": "Clear xml buffer error"}
    # try:
    #     clear_directory('/home/xml_finished')
    # except:
    #     return {"message": "Clear xml finished error"}

    return {"message": "label clear and npy folder clear"}

@app.get("/control/force/xml", tags=["control"]) # 指定 api 路徑 (get方法)
def force_xml():
    force_move_xml_files()
    return {"message": "label clear and npy folder clear"}