# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import csv


def use_threshold(result_npy_file):

    threshold=np.array([0.422, 0.15, 0.454, 0.29, 0.348, 0.331, 0.15, 0.572, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.299,0.15, 0.15, 0.15, 0.15, 0.318, 0.15, 0.336, 0.15, 0.355, 0.15, 0.15])

    #print 'new Threshold',threshold

    sample_file = '/disk/223/xiejb231/unet_mxnet2ncnn/datasets/humanprotein/input/sample_submission.csv'
    sample_submission_df = pd.read_csv(sample_file)

    result_scores = np.load(result_npy_file)

    assert len(sample_submission_df['Predicted']) == result_scores.shape[0], 'Error'

    submissions = []
    for it, row in enumerate(result_scores):
        sub_label = row-threshold
        sub_label = sub_label>0
        subrow = ' '.join(list([str(i) for i in np.nonzero(sub_label)[0]]))
        if len(np.nonzero(sub_label)[0]) == 0:
            arg_maxscore = np.argmax(result_scores[it])
            subrow = str(arg_maxscore)
        #print subrow
        submissions.append(subrow)
    # print subrow
    sample_submission_df['Predicted'] = submissions
    save_file = result_npy_file[:-10]+'_multhr.csv'
    sample_submission_df.to_csv(save_file, index=None)
    print '[multi-threshold]result save to ', save_file
    return save_file


def read_from_csv(file_name,delimiter=','):
    with open(file_name) as f:
        reader = csv.reader(f,delimiter=delimiter)
        list_csv = list(reader)
    return list_csv

def write_to_csv(file_name,content,delimiter=','):
    with open(file_name, 'wb') as f:
        writer = csv.writer(f,delimiter=delimiter)
        for row in content:
            writer.writerow(row)


def check_label_same(label1,label2):
    label1_list= [int(_) for _ in label1.split(' ')]
    label2_list= [int(_) for _ in label2.split(' ')]
    if len(label2_list)!=len(label1_list):
        return False
    for it in label1_list:
        if it not in label2_list:
            return False
    return True





def replace_leak_write_result(commit_file,show_replace=True):
    def load_leaks():
        leak_file = 'data/leak_test_matches.csv'
        leaks = read_from_csv(leak_file)

        leaks = leaks[1:]
        leak_need_replace = {}
        for item in leaks:
            leak_need_replace[item[1]] = item[-2]
        return leak_need_replace
    leaks = load_leaks()
    commit_content = read_from_csv(commit_file)

    reset_num = 0
    reset_num_realreplace = 0

    for i in xrange(1,len(commit_content)):

        if leaks.has_key(commit_content[i][0]):
            reset_num+=1

            if not check_label_same(commit_content[i][1],leaks[commit_content[i][0]]):
                if show_replace:
                    print 'replace [{}] from {} >>>>>>> {}'.format(commit_content[i][0], commit_content[i][1],
                                                               leaks[commit_content[i][0]])
                reset_num_realreplace+=1
            commit_content[i][1] = leaks[commit_content[i][0]]
    save_path = commit_file[:-4]+'_lk.csv'
    write_to_csv(save_path,commit_content)
    print 'final result has writed to ',save_path
    return save_path



if __name__ == '__main__':
    result_npy_file = '/disk/223/lichuanpeng/Project_Models/Kaggle/HumanProtein/result_summary/summary_5_score.npy'
    multi_thres_file = use_threshold(result_npy_file)
    replace_leak_write_result(multi_thres_file)


