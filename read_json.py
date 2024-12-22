import json
json_path="./predict_result_merge.json"
json_labels=json.load(open(json_path,"r"))
print(json_labels["info"])