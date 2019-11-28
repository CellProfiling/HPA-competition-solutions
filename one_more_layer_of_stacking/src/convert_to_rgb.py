import os
import cv2
import numpy as np

def channels_2_rgb_png(folder_in, folder_out, rgby=False):

    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    channels = [
            ('B', '_blue.png'),
            ('G', '_green.png'),
            ('R', '_red.png'),
        ]

    if rgby:
        channels.append(('Y', '_yellow.png'))

    def load_image(path):
        chs = []
        for c_name, c_path in channels:
            chs.append(cv2.imread(path + c_path, 0))

        im = np.stack(chs, -1)
        return im

    files_to_convert = {}
    for orig_file in os.listdir(folder_in):
        try:
            sample_id = orig_file[:orig_file.index('_')]
        except:
            print("skiping", orig_file)
            continue
        count = files_to_convert.get(sample_id, 0)
        files_to_convert[sample_id] = count + 1

    for converted_file in os.listdir(folder_out):
        sample_id = os.path.splitext(converted_file)[0]
    #     print(sample_id)
        if sample_id in files_to_convert:
            del files_to_convert[sample_id]

    for sample_id, image_count in files_to_convert.items():
        if image_count >= 3:
            im = load_image(os.path.join(folder_in, sample_id))
            cv2.imwrite( os.path.join(folder_out, sample_id) + ".png", im)
        else:
            print('skiping', sample_id)




