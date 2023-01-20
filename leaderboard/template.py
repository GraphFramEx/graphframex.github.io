import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Union
import pandas as pd 
import numpy as np

from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader



def generate_leaderboard(dataset: str,
                         models_folder: str = "leaderboard/results/topk") -> str:
    """Prints the HTML leaderboard starting from the .json results.

    The result is a <table> that can be put directly into the RobustBench index.html page,
    and looks the same as the tables that are already existing.

    The .json results must have the same structure as the following:
    ``
    {
      "link": "https://arxiv.org/abs/2003.09461",
      "name": "Adversarial Robustness on In- and Out-Distribution Improves Explainability",
      "year": 2019
      "dataset": "cora",
      "charact": "0.8",
      "fid": "0.3",
      "fidinv": "0.01"
    }
    ``


    :param dataset: The dataset of the wanted leaderboard.
    :param models_folder: The base folder of the model jsons (e.g. our "model_info" folder).

    :return: The resulting HTML table.
    """

    folder = Path(models_folder) / dataset
    print(folder)
    for model_path in folder.glob("*.json"):
        with open(model_path) as fp:
            models = json.load(fp)
    models.sort(key=lambda x: x['charact'], reverse=True)

    templateLoader = FileSystemLoader(searchpath="./")
    env = Environment(loader=templateLoader,
                      autoescape=select_autoescape(['html', 'xml']))

    template = env.get_template('leaderboard/leaderboard.html.j2')

    result = template.render(dataset=dataset, models=models)
    
    html_path = f'leaderboard/html/{dataset}.html'
    Func = open(html_path, 'w')
    Func.write(result)
    Func.close()

    return result


if __name__ == "__main__":
    
    models_folder = "leaderboard/results/topk_10"
    df = pd.read_csv('leaderboard/nc_real_topk_5expe.csv')
    list_datasets = np.unique(df.dataset)
    for dataset in list_datasets:
        generate_leaderboard(dataset, models_folder)