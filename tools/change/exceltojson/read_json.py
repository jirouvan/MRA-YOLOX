import json
json_path="./train.json"
json_labels=json.load(open(json_path,"r"))
print(json_labels["info"])