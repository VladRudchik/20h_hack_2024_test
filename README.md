# INT20h Hack 2024 test stage

### Team: OS

## Solution explanation
Для вирішення цієї задачі було використано CNN Segmentaion модель. Замість проблеми object detection на її основі (заповнюючи надані таргетом bbox як цілі сегменти) була вирішена проблема сегментації. Після чого, знайдені межі сегментів, як нові спрогнозовані bbox.

## Our Environment:
We don't know how to use Docker, so the requirement.txt is built for Windows 11 + Conda/Python 3.10.

## Install requirements:
pip install -r requirement.txt


## Inference
Inference run example:

python inference.py --test_folder_path "data/stage_2_test_images"

python inference.py --test_folder_path "data/stage_2_test_images" --result_csv_path "submission.csv"


## Train

Download data from competition and unzip.

Training run example:

python train.py --folder_image_path "data/stage_2_train_images/" --class_info_path "data/stage_2_detailed_class_info.csv" --train_labels_path "data/stage_2_train_labels.csv" --result_model_path "weights/model_weights_1.h5"
