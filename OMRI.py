import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import random
#from sklearn.preprocessing import StandardScaler
# from skbio import DistanceMatrix ####relative abundance / bray curtis
# from skbio.stats.ordination import pcoa #####pca
# from skbio.diversity import beta_diversity
import seaborn as sns
#import statannot
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import IsolationForest
import os


class Tree_Node:
    def __init__(self, depth=0, left=None, right=None, split_att=None, split_val=None, size=0, samples=[]):
        self.depth = depth
        self.left = left
        self.right = right
        self.split_att = split_att
        self.split_val = split_val
        self.size = size
        self.samples = samples


'''omriTree is represented by it's root (treeNode type)'''
class Forest:
    def __init__(self, number_of_trees=0, trees=[]):
        self.number_of_trees = number_of_trees
        self.trees = trees

    '''
        inputs: X=input data, t=number of trees, psi=sub sampling size, l=depth limit
        function: Creates forest of omriTrees.
        output: an omriForest
    '''
    def fit(self, X, psi, l, t=100, distance_matrix="de_brui"):
        for i in range(t):
            self.trees.append(omri_tree(X, 0, l, psi, distance_matrix, 0))
            self.number_of_trees += 1


'''
    inputs: X=input data, OTUs=taxa, e=current tree depth, l=depth limit, psi=sub sampling size, distance_matrix = chosen function to calcukate distance matrix
    function:
    output: an omriTree represented by it's root (treeNode type)
'''
def omri_tree(X, e, l, psi, distance_matrix=None, random_state=0):
    #np.random.seed(0)

    X = filter_features(X) # remove features with few data

    if X.empty:
        return None

    elif e >= l or len(X.iloc[0]) <= 1 or len(X) <= 1:
        return Tree_Node(depth=e, size=len(X), samples=X.columns)

    else:
        if len(X) < psi:
            sampled_X = X
        else:
            sampled_X = X.sample(n=psi, random_state=random_state) # sub-sampling of data, only psi features are remain

        sampled_X.loc[:, sampled_X.sum() > 0]

        if sampled_X.empty:
            return None

        elif len(sampled_X.iloc[0]) < 1:
            return Tree_Node(depth=e, size=len(sampled_X), samples=sampled_X.columns)

        renormalized_X = relative_abundence(sampled_X)

        if distance_matrix != "No":
            distance_mat = beta_diversity(distance_matrix, renormalized_X.T, renormalized_X.columns)
            # distance_mat.to_csv("/specific/elhanan/PROJECTS/ANOMALY_DETECTION_OP/distanceMat.txt", index=False)
            print(distance_mat.shape)

        distance_mat = renormalized_X

        #print(distance_mat_std.isnull())
        #pc1 = pcoa(distance_mat).samples[['PC1']] ####????? extract 1st pcoa

        # if (distance_mat.sum(axis=0) <= 0).any() or distance_mat.sum(axis=1).any() <= 0:
        #     print(distance_mat)
        if distance_matrix != "No":
            pc1 = pcoa(distance_mat, number_of_dimensions=1)
            print(pc1)
        else:
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(distance_mat.transpose())

        #pc1 = pca.fit_transform(distance_mat_std.transpose())

        # print("explained variance ratio", pca.explained_variance_ratio_)
        # print("var", pca.explained_variance_)

        #pca_df = pd.DataFrame(data=pc1, columns=['pc1'])
        #pc1 = PCA().fit(distance_mat).components_[0]

        # recursive calls
        t = random.uniform(min(pc1), max(pc1))
        left_pc = [i for i in range(len(pc1)) if pc1[i][0] < t]
        right_pc = [i for i in range(len(pc1)) if pc1[i][0] > t]
        X_left = X.iloc[:, left_pc]
        X_right = X.iloc[:, right_pc]
        left = omri_tree(X_left, e+1, l, psi, distance_matrix, random_state + 1)
        right = omri_tree(X_right, e+1, l, psi, distance_matrix, random_state + 1)

        return Tree_Node(depth=e, left=left, right=right, split_att=pc1, split_val=t, size=len(sampled_X.columns), samples=X.columns)


def average_depth(sample, OF):
    d = 0
    for tree in OF.trees:
        position = tree_position(sample, tree)
        d += position.depth
    d /= OF.number_of_trees
    return d


def tree_position(sample, tree):
    node = tree
    position = tree
    while node:
        if sample in node.samples:
            if node.left and sample in node.left.samples:
                node = node.left
            elif node.right and sample in node.right.samples:
                node = node.right
            else:
                position = node
                break
        else: # should never reach here
            break
    return position


'''
    inputs: A= a group of anomaly samples (A in X), B= a group of normal samples (A in X), OF= omri forest
    function:
    output: True if anomalies has lower average depth than normal samples, otherwise False
'''
def test_groups_by_depths(A, B, OF):
    A_depths = []
    for anomaly in A:
        A_depths.append(average_depth(anomaly, OF))
    B_depths = []
    for normal_sample in B:
        B_depths.append(average_depth(normal_sample, OF))
    return A_depths, B_depths

#
# def kde(title, file_name, B_depths, A_depths=[]):
#     if (A_depths != []) and (type(A_depths[0]) == list):
#         all_A = []
#         for lst in A_depths:
#             for m in lst:
#                 all_A.append(m)
#     else:
#         all_A = A_depths
#     if type(B_depths[0]) == list:
#         all_B = []
#         for lst in B_depths:
#             for m in lst:
#                 all_B.append(m)
#     else:
#         all_B = B_depths
#
#     if all_A == []:
#         sns.kdeplot(data=B_depths)
#     else:
#         sns.kdeplot(data=all_A, label='outliers')
#         sns.kdeplot(data=all_B, label='normals')
#         plt.legend()
#
#     plt.title(title)
#     plt.xlabel("depths")
#     plt.tight_layout()
#     plt.savefig(file_name, bbox_inches='tight')
#     plt.close()
#     plt.figure().clear()
#     plt.cla()


def create_data(data):
    data, OTUs = filter_samples(data)
    data = relative_abundence(data)
    return data, OTUs


def filter_samples(data):
    filtered_data_with_OTUs = data.loc[:, data.sum() > 3000]
    filtered_data = filtered_data_with_OTUs.iloc[:, 1:]
    OTUs = filtered_data_with_OTUs.iloc[:, :1]
    return filtered_data, OTUs


def relative_abundence(data):
    final_data = data.copy()
    for colname in data.columns:
        if sum(data[colname]) > 0:
            final_data.loc[:, colname] = data[colname] / sum(data[colname])
    return final_data


def filter_features(data):
    data = data[data.mean(axis=1) > 0.005]
    return data


def load_dataset(path):
    data = pd.read_csv(path,sep='\t')
    return data


def concat_data_frames(lst, axis):
    df = pd.DataFrame()
    for d in lst:
        df = pd.concat([df, d], axis=axis)
    df.fillna(value=0, inplace=True)
    return df


def write_list_to_file(file_name, lst):
    if type(lst[0]) == list:
        return write_list_of_lists_to_file(file_name, lst)
    with open(file_name + ".txt", "a") as outfile:
        # outfile.write(title + "\n")
        outfile.write("[")
        outfile.write("".join(str(item) + ", " for item in lst))
        outfile.write("]\n")
    outfile.close()

def write_list_of_lists_to_file(file_name, lst):
    with open(file_name + ".txt", "a") as outfile:
        # outfile.write(title + "\n")
        outfile.write("[")
        n = len(lst)
        for i in range(n):
            outfile.write("[")
            to_write = ""
            for j in range(len(lst[i])):
                to_write += str(lst[i][j])
                if j != len(lst[i]) - 1:
                    to_write += ", "
            outfile.write(to_write)
            if i == n-1:
                outfile.write("]")
            else:
                outfile.write("],")
        outfile.write("]\n")
    outfile.close()


def box_plot_hue(auc, auc_IF, title, dir_name):
    df = pd.DataFrame()
    auc_df = pd.DataFrame()
    auc_df["model"] = ["Omri Tree" for j in range(len(auc))]
    auc_df["auc"] = auc
    auc_df["y"] = "y1"
    df = pd.concat([df, auc_df], ignore_index=True)
    auc_IF_df = pd.DataFrame()
    auc_IF_df["model"] = ["Isolation Forest" for j in range(len(auc_IF))]
    auc_IF_df["auc"] = auc_IF
    auc_IF_df["y"] = "y2"
    df = pd.concat([df, auc_IF_df], ignore_index=True)
    # print(df)
    sns.boxplot(data=df, x="auc",y="y", hue="model").set(title=title)
    # plt.axvline(x=0.95, color='g')
    plt.xlim(0.8,1)
    plt.tight_layout()
    plt.title(title)
    plt.savefig(dir_name + '/' + title)
    plt.close()
    plt.figure().clear()
    plt.cla()


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def supervided_test(data, outliers_samples, normals_samples, l, t, psi, distance_matrix):
    # A_depths, B_depths, compare_depths, auc_lst, auc_IF, p, auc_p_r, auc_IF_p_r = [], [], [], [], [], [], [], []

    OF = Forest(0, [])
    OF.fit(data, psi, l, t, distance_matrix=distance_matrix)

    # Test
    A_depths, B_depths = test_groups_by_depths(outliers_samples, normals_samples, OF)
    compare_depths = (sum(A_depths) / len(A_depths)) < (sum(B_depths) / len(B_depths))
    U1, p = mannwhitneyu(A_depths, B_depths, alternative="less")
    y_true = len(normals_samples) * [1] + len(outliers_samples) * [0]
    auc_score = roc_auc_score(y_true, B_depths + A_depths)
    depths = B_depths + A_depths
    precision, recall, thresholds = precision_recall_curve(y_true, depths)
    auc_p_r_score = auc(recall, precision)

    # Isolation Forest test
    model = IsolationForest(n_estimators=t)
    model.fit(data.transpose())
    scores = model.decision_function(data.transpose())  # higher score means more normal
    auc_IF_score = roc_auc_score(y_true, scores)
    precision_IF, recall_IF, thresholds_IF = precision_recall_curve(y_true, scores)
    auc_IF_p_r_score = auc(recall_IF, precision_IF)

    # create_dir(dir_name)
    # df_normals.to_csv(dir_name + "/normals depths in each iteration.txt", sep='\t')
    # df_outliers.to_csv(dir_name + "/outliers depths in each iteration.txt", sep='\t')
    # write_list_to_file(dir_name + "/anomalies depths", A_depths)
    # write_list_to_file(dir_name + "/normal depths", B_depths)
    # write_list_to_file(dir_name + "/compare depths", compare_depths)
    # write_list_to_file(dir_name + "/p values", p)
    # write_list_to_file(dir_name + "/auc", auc_lst)
    # write_list_to_file(dir_name + "/auc IF", auc_IF)
    # write_list_to_file(dir_name + "/auc precision recall", auc_p_r)
    # write_list_to_file(dir_name + "/auc IF precision recall", auc_IF_p_r)

    # if times > 1:
    #     A_depths = [item for sublist in A_depths for item in sublist]
    #     B_depths = [item for sublist in B_depths for item in sublist]
    # else:
    #     A_depths = A_depths[0]
    #     B_depths = B_depths[0]
    # kde(title + "\ndepths kde", dir_name + "/kde", B_depths, A_depths)
    # box_plot_hue(auc_p_r, auc_IF_p_r, "auc Precision-Recall OMRI vs Isolation Forest", dir_name)
    return A_depths, B_depths, compare_depths, p, auc_score, auc_IF_score, auc_p_r_score, auc_IF_p_r_score

def outliers_against_normals(outliers_data, normals_data,l, t, psi, outliers_percentage, distance_matrix, dir_name, times, title):

    outliers_data = outliers_data.loc[:, outliers_data.sum() > 0] # filter samples with no data

    normals_data = normals_data.loc[:, normals_data.sum() > 0]
    normals_samples = normals_data.columns

    # outliers_count = int(outliers_percentage * len(normals_samples) / (100 - outliers_percentage))
    # l = int(math.log(len(normals_samples)+outliers_count,2)+ 20)

    A_depths, B_depths, compare_depths, p, auc_lst, auc_IF, auc_p_r, auc_IF_p_r = supervided_test(outliers_data, normals_data,l, t, psi, outliers_percentage, distance_matrix, dir_name, times, title)
    return A_depths, B_depths, compare_depths, p, auc_lst, auc_IF, auc_p_r, auc_IF_p_r


# change features to genius level only
def genius_level(str):
    i = str[::-1].find(";", 0, len(str))
    i = str[::-1].find(";", i + 1, len(str))
    return str[:len(str) - i]
#

# outliers_against_normals(healthy_vincent_data, healthy_schubert_data, l, t, psi, outliers_percentage, distance_matrix, dir_name, times, title)

def unsupervised_test(times, data, psi, l, t, distance_matrix, dir_name):
    data = relative_abundence(data)
    df = pd.DataFrame()
    samples = data.columns
    df["samples"] = samples
    all_depths = []
    for i in range(times):
        OF = Forest(0, [])
        OF.fit(data, psi, l, t, distance_matrix)
        A_depths, now_B_depths = test_groups_by_depths([], samples, OF) # A_depths = []
        df[str(i)] = now_B_depths
        all_depths += now_B_depths
    title = "unsupervised test, " + str(times) + " times " + str(t) + " trees " + str(psi) + " sub-sample  " + str(l) + " depth limit"
    df.to_csv(dir_name + "/depths in each iteration.txt", sep='\t')
    print(df)
    #test
    kde(title, dir_name + "/kde", now_B_depths, A_depths)  # A_depths = []

def heatmap_auc_scores(all_auc, all_auc_IF, all_auc_p_r, all_auc_IF_p_r, dir_name):
    all_auc = pd.pivot_table(all_auc, values="auc", index="psi percentage", columns="outliers percentage")
    all_auc_IF = pd.pivot_table(all_auc_IF, values="auc_IF", index="psi percentage", columns="outliers percentage")
    all_auc_p_r = pd.pivot_table(all_auc_p_r, values="auc_p_r", index="psi percentage", columns="outliers percentage")
    all_auc_IF_p_r = pd.pivot_table(all_auc_IF_p_r, values="auc_IF_p_r", index="psi percentage", columns="outliers percentage")

    f, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    g1 = sns.heatmap(all_auc, annot=True, cmap="YlGnBu", cbar=False, ax=axs[0,0])
    g1.set_xlabel('')
    g2 = sns.heatmap(all_auc_IF, annot=True, cmap="YlGnBu", cbar=False, ax=axs[0,1])
    g2.set_xlabel('')
    g2.set_ylabel('')
    g3 = sns.heatmap(all_auc_p_r, annot=True, cmap="YlGnBu", cbar=False, ax=axs[1,0])
    g4 = sns.heatmap(all_auc_IF_p_r, annot=True, cmap="YlGnBu", cbar=False, ax=axs[1,1])
    g4.set_ylabel('')

    axs[0,0].set_title("auc")
    axs[0,1].set_title("auc IF")
    axs[1,0].set_title("auc precision recall")
    axs[1,1].set_title("auc IF precision recal")

    plt.tight_layout()
    plt.savefig(dir_name + "/AUC scores")
    plt.close()
    plt.figure().clear()
    plt.cla()

def kde(all_A_depths, all_B_depths, outliers_percentages, psis, dir_name):
    f, axs = plt.subplots(3, 3, sharex=True, sharey=True)

    for i in range(len(psis)):
        for j in range(len(outliers_percentages)):
            sns.kdeplot(data=all_B_depths[(all_B_depths["outliers percentage"] == outliers_percentages[j]) & (all_B_depths["psi percentage"] == psis[i])]["normals depths"],
                        label='normals', ax=axs[i, j])
            sns.kdeplot(data=all_A_depths[(all_A_depths['outliers percentage'] == outliers_percentages[j]) & (all_A_depths['psi percentage'] == psis[i])]["outliers depths"],
                        label='outliers', ax=axs[i,j])

    # Add superlabels
    f.suptitle("KDE", x=0.5, y=0.995, fontsize=11)
    f.text(0.5, 0.02, 'outliers percentage', ha='center', va='center')
    f.text(0.015, 0.5, 'psi percentage', ha='center', va='center', rotation='vertical')
    f.subplots_adjust(top=0.85, bottom=0.1, left=0.1, right=0.8, hspace=0.4, wspace=0.4)

    # Add labels to the x-axis and y-axis of each subplot
    for i in range(3):
        axs[-1, i].set_xlabel(str(outliers_percentages[i]))
        axs[i, 0].set_ylabel(str(psis[i]))

    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(dir_name + "/kde depths")
    plt.close()
    plt.figure().clear()
    plt.cla()

def contamination_test(times, contamination_percentage, basic_data, contaminating_data, outliers_percentage_sample, psi, l, dir_name, different_sample):

    # filter to common features
    features_intersection = pd.Series(list(set(basic_data["#OTU ID"]).intersection(set(contaminating_data["#OTU ID"]))))
    basic_data = basic_data[basic_data["#OTU ID"].isin(features_intersection)]
    contaminating_data = contaminating_data[contaminating_data["#OTU ID"].isin(features_intersection)]

    # drop features
    basic_data = basic_data.drop("#OTU ID", axis=1)
    contaminating_data = contaminating_data.drop("#OTU ID", axis=1)

    # drop samples with no data
    basic_data = basic_data.loc[:, basic_data.sum() > 0]
    contaminating_data = contaminating_data.loc[:, contaminating_data.sum() > 0]

    # relative abundance
    basic_data = relative_abundence(basic_data)
    contaminating_data = relative_abundence(contaminating_data)

    outliers_count = int((outliers_percentage_sample * len(basic_data.columns)) / (100 - outliers_percentage_sample))

    df_outliers = pd.DataFrame()
    df_normals = pd.DataFrame()
    df_outliers_with_psi_outliers = pd.DataFrame()
    df_normals_with_psi_outliers = pd.DataFrame()

    df_scores = pd.DataFrame()
    df_auc = pd.DataFrame()
    df_auc_IF = pd.DataFrame()
    df_auc_p_r = pd.DataFrame()
    df_auc_IF_p_r = pd.DataFrame()

    for i in range(times):
        basic_data_modify = basic_data
        # sample outliers
        contaminating_data_sampled = contaminating_data.sample(n=outliers_count, axis=1)
        contaminating_data_sampled = contaminating_data_sampled.reset_index(drop=True)

        # contaminate data
        if (different_sample):
            helper_contaminating = (1 - (contamination_percentage / 100)) * basic_data_modify.sample(n=outliers_count, axis=1)
            for i in range(outliers_count):
                helper_contaminating.iloc[:,i] += (contamination_percentage / 100) * contaminating_data_sampled.iloc[:,i]
        else:
            helper_contaminating = (1 - (contamination_percentage / 100)) * basic_data_modify[contaminating_data.columns]
            helper_contaminating += (contamination_percentage / 100) * contaminating_data_sampled

        # drop contaminated samples from basic data
        samples_intersection = pd.Series(list(set(basic_data_modify.columns).intersection(set(helper_contaminating.columns))))
        a = basic_data_modify.columns.difference(samples_intersection)
        basic_data_modify = basic_data_modify[a]

        normals_samples = basic_data_modify.columns
        outliers_samples = helper_contaminating.columns
        df_outliers["samples"] = outliers_samples
        df_normals["samples"] = normals_samples
        data = concat_data_frames([basic_data_modify, helper_contaminating], 1)
        data = relative_abundence(data)

        psi_count = int((psi / 100) * len(data))
        A_depths, B_depths, compare_depths, p, auc_score, auc_IF, auc_p_r, auc_IF_p_r = supervided_test(data, outliers_samples, normals_samples, l, t, psi_count, distance_matrix)

        df_normals[str(i)] = B_depths
        df_outliers[str(i)] = A_depths

        help = pd.DataFrame({"psi percentage": psi, "outliers percentage": outliers_percentage, "normals depths": B_depths})
        df_normals_with_psi_outliers = pd.concat([df_normals_with_psi_outliers, help], axis=0)
        help = pd.DataFrame({"psi percentage": psi, "outliers percentage": outliers_percentage, "outliers depths": A_depths})
        df_outliers_with_psi_outliers = pd.concat([df_outliers_with_psi_outliers, help], axis=0)

        help = pd.DataFrame({"p": [p], "auc": [auc_score], "auc_IF": [auc_IF], "auc_p_r": [auc_p_r], "auc_IF_p_r": [auc_IF_p_r], "compare_depths": [compare_depths]})
        df_scores = pd.concat([df_scores, help], axis=0)
        help = pd.DataFrame({"psi percentage": psi, "outliers percentage": outliers_percentage, "auc": [auc_score]})
        df_auc = pd.concat([df_auc, help], axis=0)
        help = pd.DataFrame({"psi percentage": psi, "outliers percentage": outliers_percentage, "auc_IF": [auc_IF]})
        df_auc_IF = pd.concat([df_auc_IF, help], axis=0)
        help = pd.DataFrame({"psi percentage": psi, "outliers percentage": outliers_percentage, "auc_p_r": [auc_p_r]})
        df_auc_p_r = pd.concat([df_auc_p_r, help], axis=0)
        help = pd.DataFrame({"psi percentage": psi, "outliers percentage": outliers_percentage, "auc_IF_p_r": [auc_IF_p_r]})
        df_auc_IF_p_r = pd.concat([df_auc_IF_p_r, help], axis=0)


    df_normals.to_csv(dir_name + "/normals depths in each iteration.txt", sep='\t')
    df_outliers.to_csv(dir_name + "/outliers depths in each iteration.txt", sep='\t')
    df_scores.to_csv(dir_name + "/scores.txt", sep='\t')

    all_B_depths = df_normals.iloc[:, 1:].stack().tolist()
    all_A_depths = df_outliers.iloc[:, 1:].stack().tolist()
    # kde(title, dir_name + "/kde", all_B_depths, all_A_depths)

    # all_auc_p_r = df_auc_p_r["auc_p_r"]
    # all_auc_IF_p_r = df_auc_IF_p_r["auc_IF_p_r"]
    # box_plot_hue(all_auc_p_r, all_auc_IF_p_r, "auc Precision-Recall OMRI vs Isolation Forest", dir_name)
    return df_outliers_with_psi_outliers, df_normals_with_psi_outliers, df_auc, df_auc_IF, df_auc_p_r, df_auc_IF_p_r

# data must contain features column, under the name: "#OTU ID"
if len(sys.argv) == 12:
    distance_matrix = sys.argv[1]
    t = int(sys.argv[2])
    psi = int(sys.argv[3])
    outliers_percentage = int(sys.argv[4])
    l = int(sys.argv[5])
    times = int(sys.argv[6])
    basic_data = load_dataset(sys.argv[7]).iloc[:, 1:]
    contaminating_data = load_dataset(sys.argv[8]).iloc[:, 1:]
    dir_name = sys.argv[9] + " " + str(times) + " times " + str(t) + " trees " + str(
        outliers_percentage) + "% of outliers " + str(psi) + " sub sample " + str(l) + " depth limit"
    title = sys.argv[9] + "\n" + str(times) + " times " + str(t) + " trees " + str(
        outliers_percentage) + "% of outliers " + str(psi) + " sub sample " + str(l) + " depth limit"
    create_dir(dir_name)
    contamination_percentage = int(sys.argv[10])
    # basic data and contaminating data contains different sample or not
    different_sample = sys.argv[11]
    contamination_test(times, contamination_percentage, basic_data, contaminating_data,
                       outliers_percentage, psi, l, dir_name, different_sample)

# python3, distance_matrix=No, t, psi, outliers_percentage, l, time, data1(normal), data2(outliers) optional!, dir_name
if len(sys.argv) == 10:
    distance_matrix = sys.argv[1]
    t = int(sys.argv[2])
    psi = int(sys.argv[3])
    outliers_percentage = int(sys.argv[4])
    l = int(sys.argv[5])
    times = int(sys.argv[6])
    normal_data = load_dataset(sys.argv[7]).iloc[:, 1:]
    outliers_data = load_dataset(sys.argv[8]).iloc[:, 1:]
    dir_name = sys.argv[9] + str(times) + " times " + str(t) + " trees " + str(outliers_percentage) + "% of outliers " + str(psi) + " sub sample " + str(l) + " depth limit"
    title = sys.argv[9] + "\n" + str(times) + " times " + str(t) + " trees " + str(outliers_percentage) + "% of outliers " + str(psi) + " sub sample " + str(l) + " depth limit"
    outliers_against_normals(outliers_data, normal_data, l, t, psi, outliers_percentage,distance_matrix,dir_name, times, title)

# # python3, distance_matrix=None, t, psi, l, times, data, dir_name
# elif len(sys.argv) == 8:
#     distance_matrix = sys.argv[1]
#     t = int(sys.argv[2])
#     psi = int(sys.argv[3])
#     l = int(sys.argv[4])
#     times = int(sys.argv[5])
#     data = load_dataset(sys.argv[6]).iloc[:, 2:]
#     dir_name = sys.argv[7] + " " + str(times) + " times " + str(t) + " trees " + str(psi) + " sub sample " + str(l) + " depth limit"
#     create_dir(dir_name)
#     unsupervised_test(times, data, psi, l, t, distance_matrix, dir_name)




# matrix maker for this parameters: psis = [10, 30, 50], outliers_percentages = [1,3,5]
elif len(sys.argv) == 8:
    distance_matrix = sys.argv[1]
    t = 50
    psis = [10, 30, 50]
    outliers_percentages = [1,3,5]
    times = int(sys.argv[2])
    basic_data = load_dataset(sys.argv[3]).iloc[:, 1:]
    contaminating_data = load_dataset(sys.argv[4]).iloc[:, 1:]
    contamination_percentage = int(sys.argv[5])
    l = len(basic_data.iloc[:,1])
    name = sys.argv[6]
    # basic data and contaminating data contains different sample or not
    different_sample = sys.argv[7]

    all_A_depths = pd.DataFrame()
    all_B_depths = pd.DataFrame()
    all_auc = pd.DataFrame()
    all_auc_IF = pd.DataFrame()
    all_auc_p_r = pd.DataFrame()
    all_auc_IF_p_r = pd.DataFrame()
    for psi_perc in psis:
        for outliers_percentage in outliers_percentages:
            title_help = str(times) + " times " + str(contamination_percentage) + "% contamination " + str(t) + " trees " + str(
                outliers_percentage) + "% of outliers " + str(psi_perc) + "% sub sample " + str(l) + " depth limit"

            create_dir(name)
            dir_name = name + "/" + title_help
            title = name + "\n" + title_help
            create_dir(dir_name)

            A_depths, B_depths, auc_score, auc_IF, auc_p_r, auc_IF_p_r = contamination_test(times, contamination_percentage, basic_data, contaminating_data, outliers_percentage,
                               psi_perc, l, dir_name, different_sample)
            all_A_depths = pd.concat([A_depths, all_A_depths], axis=0)
            all_B_depths = pd.concat([B_depths, all_B_depths], axis=0)
            all_auc = pd.concat([auc_score, all_auc], axis=0)
            all_auc_IF = pd.concat([auc_IF, all_auc_IF], axis=0)
            all_auc_p_r = pd.concat([auc_p_r, all_auc_p_r], axis=0)
            all_auc_IF_p_r = pd.concat([auc_IF_p_r, all_auc_IF_p_r], axis=0)

    all_auc.to_csv(name + "/all_auc")
    all_auc_IF.to_csv(name + "/all_auc_IF")
    all_auc_p_r.to_csv(name + "/all_auc_p_r")
    all_auc_IF_p_r.to_csv(name + "/all_auc_IF_p_r")

    heatmap_auc_scores(all_auc, all_auc_IF, all_auc_p_r, all_auc_IF_p_r, name)

    kde(all_A_depths, all_B_depths, outliers_percentages, psis, name)





