import os
import json

def write_json(dict,out_folder):
    file=os.path.join(out_folder,'simulation_report.json')
    with open(file, 'w') as fp:
        json.dump(dict, fp, indent=4)

def read_json(folder):
    file=os.path.join(folder,'simulation_report.json')
    with open(file,'r') as fp:
        report=json.load(fp)
    return report