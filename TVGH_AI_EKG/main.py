from fastapi import FastAPI, File, UploadFile
from my_inference import ai_test
from xml2npy_filter import decode_and_filter
from lvsdinf.lvsdclassifier import LVSDClassifier
import numpy as np
import pandas as pd
from csvCreater import create_csv
from fastapi.middleware.cors import CORSMiddleware
import os

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
    diagnosis = ['HRrEF', 'HFpEF', 'Normal']

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

@app.get("/control/clear", tags=["control"]) # 指定 api 路徑 (get方法)
def clear_csv_npy():
    create_csv()
    directory_to_clear = '/home/npy'
    clear_directory(directory_to_clear)
    
    return {"message": "label clear and npy folder clear"}