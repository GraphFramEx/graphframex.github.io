import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

from jinja2 import Environment, PackageLoader, select_autoescape



def generate_leaderboard(dataset: str,
                         models_folder: str = "model_info") -> str:
    """Prints the HTML leaderboard starting from the .json results.

    The result is a <table> that can be put directly into the RobustBench index.html page,
    and looks the same as the tables that are already existing.

    The .json results must have the same structure as the following:
    ``
    {
      "link": "https://arxiv.org/abs/2003.09461",
      "name": "Adversarial Robustness on In- and Out-Distribution Improves Explainability",
      "dataset": "cifar10",
      "eps": "0.5",
      "charact": "0.8",
      "fid": "0.3",
      "fidinv": "0.01"
    }
    ``


    :param dataset: The dataset of the wanted leaderboard.
    :param models_folder: The base folder of the model jsons (e.g. our "model_info" folder).

    :return: The resulting HTML table.
    """

    folder = Path(models_folder)
    for model_path in folder.glob("*.json"):
        with open(model_path) as fp:
            models = json.load(fp)
    models.sort(key=lambda x: x['charact'], reverse=True)

    env = Environment(loader=PackageLoader('graphframex', 'leaderboard'),
                      autoescape=select_autoescape(['html', 'xml']))

    template = env.get_template('leaderboard.html.j2')

    result = template.render(dataset=dataset, models=models)
    print(result)
    return result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="The dataset of the desired leaderboard."
    )
    parser.add_argument(
        "--models_folder",
        type=str,
        default="model_info",
        help="The base folder of the model jsons (e.g. our 'model_info' folder)"
    )
    args = parser.parse_args()

    generate_leaderboard(args.dataset, args.models_folder)