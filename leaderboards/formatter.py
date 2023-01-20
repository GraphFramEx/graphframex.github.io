import pandas as pd
import csv 
import json 
import numpy as np
import os

def csv_to_json(csvFilePath, jsonFilePath):
    jsonArray = []
      
    #read csv file
    with open(csvFilePath, encoding='utf-8') as csvf: 
        #load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csvf) 

        #convert each csv row into python dict
        for row in csvReader: 
            #add this python dict to json array
            jsonArray.append(row)
  
    #convert python jsonArray to JSON String and write to file
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf: 
        jsonString = json.dumps(jsonArray, indent=4)
        jsonf.write(jsonString)

def inform_csv(dataset, topk=10):
    df = pd.read_csv('nc_real_topk_5expe.csv')
    df_sub = df[(df.dataset==dataset)&(df.topk==topk)]

    df_infos = pd.read_csv('infos.csv')
    df_sub.join(df_infos, on='explainer_name')
    print(df_sub)
    df_sub = df_sub.reset_index()
    print(df_sub.columns)
    #df_sub.to_csv(f'results/{topk}/{dataset}_topk_{topk}_res.csv', index=False)
    return


if __name__ == "__main__":
    topk = 10
    os.makedirs('results/{}'.format(topk), exist_ok=True)
    df = pd.read_csv('nc_real_topk_5expe.csv')
    list_datasets = np.unique(df.dataset)
    
    """for dataset in list_datasets:
        inform_csv(dataset, topk=topk)
        csvFilePath =  f'results/{topk}/{dataset}_topk_{topk}_res.csv'
        jsonFilePath = f'results/{topk}/{dataset}_topk_{topk}_res.json'
        csv_to_json(csvFilePath, jsonFilePath)
"""