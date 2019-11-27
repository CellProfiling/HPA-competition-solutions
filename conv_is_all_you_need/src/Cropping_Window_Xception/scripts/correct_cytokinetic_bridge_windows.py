import pandas as pd
from tqdm import tqdm
import json
from lib.utils import str_to_labels, labels_to_str, debug, multiprocessing

with open('data/marker_list.json', 'r') as f:
    correction_dict = json.load(f)

classes_to_correct = [16, 17]


def window_contains_rect(window, rect):
    left, top, right, bottom = window
    r_left, r_top, r_right, r_bottom = rect

    # debug(f'(left, top, right, bottom) = {(left, top, right, bottom)}')
    # debug(f'(r_left, r_top, r_right, r_bottom) = {(r_left, r_top, r_right, r_bottom)}')

    return r_left >= left and r_top >= top and r_right <= right and r_bottom <= bottom


def get_corrected_target_for_one_row(task):
    id_, row = task
    labels = str_to_labels(row['Target'])
    if all([x not in labels for x in classes_to_correct]):
        return {'id_': id_, 'corrected_target': row['Target']}

    for class_to_correct in classes_to_correct:
        if class_to_correct not in labels:
            continue

        rects = correction_dict[f"markerListForClass{class_to_correct}"].get(row['source_img_id'])
        if rects is None or not any(
            [window_contains_rect((row[['left', 'top', 'right', 'bottom']]), rect) for rect in rects]
        ):
            debug(f"removed {class_to_correct} from {id_} ({labels})")
            labels.remove(class_to_correct)

    return {'id_': id_, 'corrected_target': labels_to_str(labels)}


# TODO: Weird! why is the hpa part much slower than the original part??
def main():
    anno = pd.read_csv('data/train_with_hpa.csv', index_col=0)
    anno_windowed = pd.read_csv('data/train_windowed_0.4.csv', index_col=0)
    anno_windowed = anno_windowed.join(anno[['Target']], how='left', on=['source_img_id'])

    corrected_targets = []
    for result in multiprocessing(get_corrected_target_for_one_row, anno_windowed.iterrows(), len_=len(anno_windowed)):
        corrected_targets.append(result['corrected_target'])

    anno_windowed['corrected_target'] = corrected_targets
    anno_windowed.drop('Target', axis=1).to_csv('data/train_windowed_corrected_0.4.csv')


if __name__ == '__main__':
    main()
