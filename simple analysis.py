import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp

def profile(df, name, pandas_profiling_output_path):
    profile = pp.ProfileReport(df)

    if not os.path.exists(os.path.join(pandas_profiling_output_path)):
        os.makedirs(os.path.join(pandas_profiling_output_path))

    profile.to_file(os.path.join(pandas_profiling_output_path, f'{name}_pandas_profile.html'))

dir_AOB_data = pd.read_csv(r"S:\Data_Science\Core\FDA_submission_10_2019\08-Reports\STAR_reports\Labeling_project\datasets\AOB_dataset.csv")
dir_VGNG_data = pd.read_csv(r"S:\Data_Science\Core\FDA_submission_10_2019\08-Reports\STAR_reports\Labeling_project\datasets\VGNG_dataset.csv")
dir_p3a = pd.read_csv(r"S:\Data_Science\Core\FDA_submission_10_2019\06-Data\02-Preprocessed_data\2020-02-18\AOB\AOB_Novel.csv")
dir_p3b = pd.read_csv(r"S:\Data_Science\Core\FDA_submission_10_2019\06-Data\02-Preprocessed_data\2020-02-18\AOB\AOB_Target.csv")

features_p3a = ['P3a_Delta_Novel_similarity_spatial',
            'P3a_Delta_Novel_similarity_locationLR',
            'P3a_Delta_Novel_similarity_locationPA',
            'P3a_Delta_Novel_similarity_timing',
            'P3a_Delta_Novel_similarity_amplitude',
            'P3a_Delta_Novel_matchScore',
            'P3a_Delta_Novel_attr_timeMSfromTriger',
            'P3a_Delta_Novel_attr_leftRight',
            'P3a_Delta_Novel_attr_posteriorAnterior',
            'P3a_Delta_Novel_attr_amplitude',
            'P3a_Delta_Novel_topo_topographicCorrCoeffAligned',
            'P3a_Delta_Novel_topo_topographicSimilarity']

features_p3b = ['P3b_Delta_Target_similarity_spatial',
             'P3b_Delta_Target_similarity_locationLR',
             'P3b_Delta_Target_similarity_locationPA',
             'P3b_Delta_Target_similarity_timing',
             'P3b_Delta_Target_similarity_amplitude',
             'P3b_Delta_Target_matchScore',
             'P3b_Delta_Target_attr_timeMSfromTriger',
             'P3b_Delta_Target_attr_leftRight',
             'P3b_Delta_Target_attr_posteriorAnterior',
             'P3b_Delta_Target_attr_amplitude',
             'P3b_Delta_Target_topo_topographicCorrCoeffAligned',
             'P3b_Delta_Target_topo_topographicSimilarity']
if __name__=="__main__":


    # profile(dir_AOB_data,  "AOB_dataset", r"S:\Data_Science\Core\FDA_submission_10_2019\08-Reports\STAR_reports\Labeling_project\datasets")
    # profile(dir_VGNG_data,  "AOB_dataset", r"S:\Data_Science\Core\FDA_submission_10_2019\08-Reports\STAR_reports\Labeling_project\datasets")
    # profile(dir_p3a[features_p3a],  "P3a_dataset", r"C:\Users\nogag\aob-miscatch-detection\pandas_profiling")
    # profile(dir_p3b[features_p3b],  "P3b_dataset", r"C:\Users\nogag\aob-miscatch-detection\pandas_profiling")

    df_P3a = dir_AOB_data[['taskData._id.$oid', 'miscatch_P3a_novel']].merge(dir_p3a[features_p3a[:6] + ['taskData.elm_id']],
                  left_on="taskData._id.$oid", right_on='taskData.elm_id')
    sns_plot = sns.pairplot(df_P3a, hue='miscatch_P3a_novel')
    sns_plot.savefig(os.path.join(r"C:\Users\nogag\aob-miscatch-detection\pandas_profiling", "P3a_variables_first6.png"))

    df_P3a = dir_AOB_data[['taskData._id.$oid', 'miscatch_P3a_novel']].merge(dir_p3a[features_p3a[6:] + ['taskData.elm_id']],
                  left_on="taskData._id.$oid", right_on='taskData.elm_id')
    sns_plot = sns.pairplot(df_P3a, hue='miscatch_P3a_novel')
    sns_plot.savefig(os.path.join(r"C:\Users\nogag\aob-miscatch-detection\pandas_profiling", "P3a_variables_last6.png"))

    df_P3b = dir_AOB_data[['taskData._id.$oid', 'miscatch_P3a_novel']].merge(dir_p3b[features_p3b[:6] + ['taskData.elm_id']],
                  left_on="taskData._id.$oid", right_on='taskData.elm_id')
    sns_plot = sns.pairplot(df_P3b, hue='miscatch_P3a_novel')
    sns_plot.savefig(os.path.join(r"C:\Users\nogag\aob-miscatch-detection\pandas_profiling", "P3b_variables_first6.png"))

    df_P3b = dir_AOB_data[['taskData._id.$oid', 'miscatch_P3a_novel']].merge(dir_p3b[features_p3b[6:] + ['taskData.elm_id']],
                  left_on="taskData._id.$oid", right_on='taskData.elm_id')
    sns_plot = sns.pairplot(df_P3b, hue='miscatch_P3a_novel')
    sns_plot.savefig(os.path.join(r"C:\Users\nogag\aob-miscatch-detection\pandas_profiling", "P3b_variables_last6.png"))