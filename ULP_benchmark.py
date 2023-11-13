#!/usr/bin/env python

import sys
sys.path.append('../')

import os
import pandas as pd
import numpy as np
import Levenshtein as lev


input_dir = '../../logs/' # The input directory of log file
output_dir = "./output"

benchmark_settings = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']
        },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>', 
        'regex': [r'(\d+\.){3}\d+']
        },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+']

        },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?']
     
        },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+']
      
        },

    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'regex': [r'=\d+']

        },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+']
      
        },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'regex': [r'0x.*?\s']   
        },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}']
      
        },

    'Android': {
        'log_file': 'Android/Android_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b']
 
        },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'regex': []

        },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'regex': [r'(\d+\.){3}\d+']
      
        },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B']

        },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+']

        },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+']

        },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'regex': [r'([\w-]+\.){2,}[\w-]+']
 
        },
}


def main():
    bechmark_result = []
    for dataset, setting in benchmark_settings.iteritems():
        print('\n=== Evaluation on %s ==='%dataset)
        indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
        log_file = os.path.basename(setting['log_file'])

        parser = ULP.LogParser(log_format=setting['log_format'], indir=indir, outdir=output_dir, rex=setting['regex'])
        parser.parse(log_file)

        F1_measure, accuracy = exec_main.evaluate(
                               groundtruth=os.path.join(indir, log_file + '_structured.csv'),
                               parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
                               )
        # New evaluation for Message-level accuracy and edit distance
        msg_accuracy, edit_distance_mean, edit_distance_std = evaluate_message_level(
            groundtruth=os.path.join(indir, log_file + '_structured_corrected.csv'),
            parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
        )
        bechmark_result.append([dataset, F1_measure, accuracy, msg_accuracy, edit_distance_mean, edit_distance_std])

    print('\n=== Overall evaluation results ===')
    df_result = pd.DataFrame(bechmark_result, columns=['Dataset', 'F1_measure', 'Accuracy', 'Msg_Accuracy', 'Edit_Distance_Mean', 'Edit_Distance_Std'])
    df_result.set_index('Dataset', inplace=True)
    print(df_result)
    df_result.T.to_csv('ULP_bechmark_result.csv')

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

        print('Message-Level Accuracy: %.4f, Edit Distance Mean: %.4f, Edit Distance Std: %.4f' % (
            msg_accuracy, edit_distance_mean, edit_distance_std))

        return msg_accuracy, edit_distance_mean, edit_distance_std


if __name__ == '__main__':
    main()
