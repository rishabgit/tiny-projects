from detecto.utils import read_image
import numpy as np
from scipy.spatial import distance
from mean_average_precision import MetricBuilder
from detecto import core
import pandas as pd
import json 
from tqdm import tqdm

print('Setting up test data')
test = pd.read_csv('ShelfImages/val.csv')

test['class_id'] = 0
test['difficult'] = 0
test['crowd'] = 0
test = test.drop('image_id', 1)

print('Loading model weights from saved_models/ directory')
model = core.Model.load('saved_models/model_weights.pth', ['Product'])
metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)

mAP = []
precision = []
recall = []
image2products = {}

THRESHOLD = 0.8

print('Progress-- ')
for image_name in tqdm(test['filename'].unique()):
  count_products = 0
  test_preds = []
  test_gt = []
  image = read_image(f'ShelfImages/images/{image_name}')
  preds = model.predict(image)
  for label, box, score in zip(*preds):
    if score > THRESHOLD:
      count_products += 1
      test_preds.append([box[0].item(), box[1].item(), box[2].item(), box[3].item(), 0, score])
  image2products[image_name] = count_products
  temp = test.to_numpy()
  temp = temp[temp[:, 0] == image_name]
  test_gt = temp[:,4:]
  test_preds = np.array(test_preds)
  metric_fn.add(test_preds, test_gt)
  mAP.append(metric_fn.value(iou_thresholds=0.5)['mAP'])
  temp = metric_fn.value(iou_thresholds=0.5)[0.5][0]['precision']
  precision.append(sum(temp)/len(temp))
  temp = metric_fn.value(iou_thresholds=0.5)[0.5][0]['recall']
  recall.append(sum(temp)/len(temp))

results = {'mAP':sum(mAP) / len(mAP), 'precision':sum(precision) / len(precision),\
           'recall':sum(recall) / len(recall)}

with open("image2products.json", "w") as outfile:
    json.dump(image2products, outfile)

with open("metrics.json", "w") as outfile:
    json.dump(results, outfile)