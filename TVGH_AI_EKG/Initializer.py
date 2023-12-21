import pandas as pd
from pathlib import Path
import os

def create_csv():
    label = {
        'id': [],
        'date': [],
        'diagnosis': []
    }

    Model_pred = {
        'id': [],
        'date': [],
        'sddd_model_pred': [],
        'sddd_model_prob_sd': [],
        'sddd_model_prob_dd': [],
        'sddd_model_prob_others': [],
        'sd_model_pred': [],
        'sd_model_prob_sd': [],
        'sd_model_prob_others': []
    }

    TurnOn = {
        'id': [],
        'Switch': []
    }

    df1 = pd.DataFrame(label)
    df2 = pd.DataFrame(Model_pred)
    df3 = pd.DataFrame(TurnOn)
    
    df1.to_csv('/home/label.csv', index=False)
    df2.to_csv('/home/Diagnosis.csv', index=False)
    df3.to_csv('/home/TurnOn.csv', index=False)


def create_folder():
    folder_names = ['/home/npy', '/home/xml_buffer', '/home/xml_finished']

    for folder in folder_names:
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass

def create_json():
    file_path = '/home/Patient.json'
    Path(file_path).write_text('{}')

if __name__ == '__main__':
    create_csv()
    create_folder()
    create_json()