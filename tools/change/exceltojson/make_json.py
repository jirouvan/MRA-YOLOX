def coco(df):
    annotion_id = 0
    images = []
    annotations = []

    categories = [{'id': 1, 'name': 'hopper'},
                  {'id': 2, 'name': 'spider'},
                  ]

    for i, row in enumerate(df.iterrows()):

        images.append({
            "id": i,
            "file_name": f"{row[1]['filename']}",
            "height": row[1]['photoheight'],
            "width": row[1]['photowidth'],
        })

        annotations.append({
            "id": annotion_id,
            "image_id": i,
            "category_id": row[1]['attribution'],
            "bbox": [row[1]['xmin'],row[1]['ymin'],row[1]['width'] , row[1]['height']],
            "area": row[1]['width'] * row[1]['height'],
            "segmentation": [],
            "iscrowd": 0
        })
        annotion_id += 1

    json_file = {'categories': categories, 'images': images, 'annotations': annotations}
    return json_file

import pandas as pd
import json
df_train = pd.read_excel('./train.xlsx')
df_val = pd.read_excel('./val.xlsx')



json_file_train = coco(df_train)
json_file_val = coco(df_val)

with open('./my_train.json','w',encoding='utf-8') as f:
    json.dump(json_file_train,f,ensure_ascii=True,indent=4)
with open('./my_val.json', 'w', encoding='utf-8') as f:
    json.dump(json_file_val, f, ensure_ascii=True, indent=4)