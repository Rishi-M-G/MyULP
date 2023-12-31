

import sys
import pandas as pd
import numpy as np
import Levenshtein as lev
from collections import defaultdict
try:
    from scipy.misc import comb
except ImportError as e:
    from scipy.special import comb

def evaluate(groundtruth, parsedresult):
    """ Evaluation function to org_benchmark log parsing accuracy
    
    Arguments
    ---------
        groundtruth : str
            file path of groundtruth structured csv file 
        parsedresult : str
            file path of parsed structured csv file

    Returns
    -------
        f_measure : float
        accuracy : float
    """ 
    df_groundtruth = pd.read_csv(groundtruth)
    df_parsedlog = pd.read_csv(parsedresult)
    # Remove invalid groundtruth event Ids
    null_logids = df_groundtruth[~df_groundtruth['EventId'].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedlog = df_parsedlog.loc[null_logids]
    (precision, recall, f_measure, accuracy) = get_accuracy(df_groundtruth['EventId'], df_parsedlog['EventId'])

    # New evaluation for Message-level accuracy and edit distance
    msg_accuracy, edit_distance_mean, edit_distance_std = evaluate_message_level(
        groundtruth=groundtruth,
        parsedresult=parsedresult
    )

    print('Precision: %.4f, Recall: %.4f, F1_measure: %.4f, Grouping_Accuracy (GA): %.4f, '
          'Msg_Accuracy: %.4f, Edit_Distance_Mean: %.4f, Edit_Distance_Std: %.4f'
          % (precision, recall, f_measure, accuracy, msg_accuracy, edit_distance_mean, edit_distance_std))
    return f_measure, accuracy, msg_accuracy, edit_distance_mean, edit_distance_std

def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    """ Compute accuracy metrics between log parsing results and ground truth
    
    Arguments
    ---------
        series_groundtruth : pandas.Series
            A sequence of groundtruth event Ids
        series_parsedlog : pandas.Series
            A sequence of parsed event Ids
        debug : bool, default False
            print error log messages when set to True

    Returns
    -------
        precision : float
        recall : float
        f_measure : float
        accuracy : float
    """
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    real_pairs = 0
    for count in series_groundtruth_valuecounts:
        if count > 1:
            real_pairs += comb(count, 2)

    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    parsed_pairs = 0
    for count in series_parsedlog_valuecounts:
        if count > 1:
            parsed_pairs += comb(count, 2)

    accurate_pairs = 0
    accurate_events = 0 # determine how many lines are correctly parsed
    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
        error_eventIds = (parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
        error = True
        if series_groundtruth_logId_valuecounts.size == 1:
            groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
                error = False
        if error and debug:
            print('(parsed_eventId, groundtruth_eventId) =', error_eventIds, 'failed', logIds.size, 'messages')
        for count in series_groundtruth_logId_valuecounts:
            if count > 1:
                accurate_pairs += comb(count, 2)

    precision = float(accurate_pairs) / parsed_pairs
    recall = float(accurate_pairs) / real_pairs
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_groundtruth.size
    return precision, recall, f_measure, accuracy

def evaluate_message_level(groundtruth, parsedresult):
    df_groundtruth = pd.read_csv(groundtruth)
    df_parsedlog = pd.read_csv(parsedresult, index_col=False)

    # Assuming 'EventTemplate' is the relevant column for message-level evaluation
    msg_groundtruth = df_groundtruth['EventTemplate'].values.astype('str')
    msg_parsedlog = df_parsedlog['EventTemplate'].values.astype('str')

    # Message-level accuracy
    msg_accuracy = np.mean(msg_groundtruth == msg_parsedlog)

    # Edit distance
    edit_distance_result = np.array([lev.distance(i, j) for i, j in zip(msg_groundtruth, msg_parsedlog)])
    edit_distance_mean = np.mean(edit_distance_result)
    edit_distance_std = np.std(edit_distance_result)

    print('Message-Level Accuracy: %.4f, Edit Distance Mean: %.4f, Edit Distance Std: %.4f'
          % (msg_accuracy, edit_distance_mean, edit_distance_std))

    return msg_accuracy, edit_distance_mean, edit_distance_std







