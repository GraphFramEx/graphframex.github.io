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

def add_metrics(df_res):
    df_res["1-fidelity_acc-"] = 1- df_res["fidelity_acc-"]
    df_res["1-fidelity_prob-"] = 1- df_res["fidelity_prob-"]
    df_res['charact_prob'] = 2 * df_res['fidelity_prob+'] * (1-df_res['fidelity_prob-'])/ (df_res['fidelity_prob+']  + 1-df_res['fidelity_prob-'])
    df_res['charact_acc'] = 2 * df_res['fidelity_acc+'] * (1-df_res['fidelity_acc-'])/ (df_res['fidelity_acc+']  + 1-df_res['fidelity_acc-'])

    df_res["1-fidelity_gnn_acc-"] = 1- df_res["fidelity_gnn_acc-"]
    df_res["1-fidelity_gnn_prob-"] = 1- df_res["fidelity_gnn_prob-"]
    df_res['charact_gnn_prob'] = 2 * df_res['fidelity_gnn_prob+'] * (1-df_res['fidelity_gnn_prob-'])/ (df_res['fidelity_gnn_prob+']  + 1-df_res['fidelity_gnn_prob-'])
    df_res['charact_gnn_acc'] = 2 * df_res['fidelity_gnn_acc+'] * (1-df_res['fidelity_gnn_acc-'])/ (df_res['fidelity_gnn_acc+']  + 1-df_res['fidelity_gnn_acc-'])

    df_res.loc[df_res['true_label_as_target']==False, 'charact_prob'] = df_res['charact_gnn_prob']
    df_res.loc[df_res['true_label_as_target']==False, 'fidelity_prob+'] = df_res['fidelity_gnn_prob+']
    df_res.loc[df_res['true_label_as_target']==False, "1-fidelity_prob-"] = df_res["1-fidelity_gnn_prob-"]
    df_res.loc[df_res['true_label_as_target']==False, "fidelity_prob-"] = df_res["fidelity_gnn_prob-"]

    df_res.loc[df_res['true_label_as_target']==False, 'charact_acc'] = df_res['charact_gnn_acc']
    df_res.loc[df_res['true_label_as_target']==False, 'fidelity_acc+'] = df_res['fidelity_gnn_acc+']
    df_res.loc[df_res['true_label_as_target']==False, "1-fidelity_acc-"] = df_res["1-fidelity_gnn_acc-"]
    df_res.loc[df_res['true_label_as_target']==False, "fidelity_acc-"] = df_res["fidelity_gnn_acc-"]
    return df_res

def filter(df, focus='phenomenon', hard_mask=True, topk=10):
    if focus=='phenomenon':
        df_sub = df[(df.true_label_as_target==True)&(df.hard_mask==hard_mask)&(df.topk==topk)]
        df_sub = df_sub.rename(columns={"charact_acc": "charact", "fidelity_acc+": "fid", "fidelity_acc-": "fidinv"})
        df_sub = df_sub[['explainer_name', 'dataset', 'topk', 'charact', 'fid', 'fidinv']]
    elif focus=='model':
        df_sub = df[(df.true_label_as_target==False)&(df.hard_mask==hard_mask)&(df.topk==topk)]
        df_sub = df_sub.rename(columns={"charact_acc": "charact", "fidelity_acc+": "fid", "fidelity_acc-": "fidinv"})
        df_sub = df_sub[['explainer_name', 'dataset', 'topk', 'charact', 'fid', 'fidinv']]
    else:
        raise ValueError("Unknown focus of the explanation")
    df_sub[['charact','fid','fidinv']] = df_sub[['charact','fid','fidinv']].apply(lambda x: round(x, 3))
    return df_sub



def inform_csv(df, df_infos, dataset, topk=10):
    df_sub = df[(df.dataset==dataset)&(df.topk==topk)]
    df_sub = df_sub.merge(df_infos, on='explainer_name')
    df_sub = df_sub.reset_index()
    df_sub.to_csv(f'leaderboard/results/topk_{topk}/{dataset}/{dataset}_topk_{topk}_res.csv', index=False)
    return


if __name__ == "__main__":
    topk = 10
    os.makedirs(f'leaderboard/results/topk_{topk}', exist_ok=True)
    df = pd.read_csv('leaderboard/nc_real_topk_5expe.csv')
    df = add_metrics(df)
    df = filter(df, topk=10)
    df_infos = pd.read_csv('leaderboard/infos.csv')
    list_datasets = np.unique(df.dataset)
    for dataset in list_datasets:
        os.makedirs(f'leaderboard/results/topk_{topk}/{dataset}', exist_ok=True)
        inform_csv(df, df_infos, dataset, topk=topk)
        csvFilePath =  f'leaderboard/results/topk_{topk}/{dataset}/{dataset}_topk_{topk}_res.csv'
        jsonFilePath = f'leaderboard/results/topk_{topk}/{dataset}/{dataset}_topk_{topk}_res.json'
        csv_to_json(csvFilePath, jsonFilePath)
