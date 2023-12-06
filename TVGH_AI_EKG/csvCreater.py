import pandas as pd

def create_csv():
    data = {
        'id': [],
        'date': [],
        'diagnosis': []
    }

    df = pd.DataFrame(data)
    df.to_csv('/home/label.csv', index=False)