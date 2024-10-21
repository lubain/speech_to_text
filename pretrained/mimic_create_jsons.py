import os
import json
import random
from argparse import Namespace

def main(args):
    data = []
    directory = args.file_folder_directory
    filetxtname = os.path.basename(directory)
    percent = args.percent
    
    metadata_path = os.path.join(directory, filetxtname + "gasy.txt")
    
    print(f"Metadata path: {metadata_path}")
    
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path) as f:
        for line in f:
            file_name = line.partition('|')[0]
            text = line.split('|')[1]
            data.append({
                "key": os.path.join(directory, file_name),
                "text": text.strip()
            })

    random.shuffle(data)
    
    train_json_path = os.path.join(args.save_json_path, "train.json")
    test_json_path = os.path.join(args.save_json_path, "test.json")
    
    os.makedirs(args.save_json_path, exist_ok=True)
    
    print(f"Saving train data to: {train_json_path}")
    with open(train_json_path, 'w', encoding="utf-8") as f:
        d = len(data)
        f.write("[\n")
        for i in range(int(d - d / percent)):
            line = json.dumps(data[i])
            f.write(line + ",\n")
        f.write("\n]")
    
    print(f"Saving test data to: {test_json_path}")
    with open(test_json_path, 'w', encoding="utf-8") as f:
        d = len(data)
        f.write("[\n")
        for i in range(int(d - d / percent), d):
            line = json.dumps(data[i])
            f.write(line + ",\n")
        f.write("\n]")

# Utiliser un dictionnaire pour simuler les arguments
args_dict = {
    'file_folder_directory': '',  # Assurez-vous que ce répertoire est correct et existe
    'save_json_path': 'output',  # Assurez-vous que ce répertoire est correct et existe ou peut être créé
    'percent': 10  # Vous pouvez ajuster ce pourcentage selon vos besoins
}

# Convertir le dictionnaire en un Namespace
args = Namespace(**args_dict)

# Exécuter la fonction principale avec les arguments
main(args)
