from tqdm import tqdm
import numpy as np
import shutil
import pandas as pd
import ast
import os
from sklearn.model_selection import train_test_split

data_path = "data"
output_path = "data/preprocessed"


def process_data(df: pd.DataFrame, data_type='train'):
    '''
    - Format the given bounding box into yolo format
    - Create directory structure for images and labels for yolo
    - Move the images and labels into their ideal path

    '''
    for idx, raw in tqdm(df.iterrows(), total=len(df)):
        img_id = raw['image_id']
        bboxes = raw['bboxes']
        yolo_data = []

        for bbox in bboxes:
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            x_center = x+w / 2
            y_center = y+h / 2
            x_center /= 1024
            y_center /= 1024
            w /= 1024
            h /= 1024
            yolo_data.append([0, x_center, y_center, w, h])
        yolo_data = np.array(yolo_data)

        os.makedirs(os.path.join(
            output_path, f"labels/{data_type}"), exist_ok=True)
        os.makedirs(os.path.join(
            output_path, f"images/{data_type}"), exist_ok=True)

        np.savetxt(os.path.join(
            output_path, f"labels/{data_type}/{img_id}.txt"), yolo_data, fmt=["%d", "%f", "%f", "%f", "%f"])
        shutil.copyfile(
            os.path.join(data_path, f"train/{img_id}.jpg"),
            os.path.join(output_path, f"images/{data_type}/{img_id}.jpg")
        )


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    df.bbox = df.bbox.apply(ast.literal_eval)
    df = df.groupby('image_id')['bbox'].apply(
        list).reset_index(name='bboxes')

    train_df, valid_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True)

    process_data(train_df, data_type="train")
    process_data(valid_df, data_type="valid")
