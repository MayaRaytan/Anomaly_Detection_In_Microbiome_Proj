import math
import numpy as np
import pandas as pd
# from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
#from sklearn.preprocessing import StandardScaler
from skbio import DistanceMatrix ####relative abundance / bray curtis
from skbio.stats.ordination import pcoa #####pca
from skbio.diversity import beta_diversity
import seaborn as sns
#import statannot
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import auc
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
    def fit(self, X, psi, l, t=100, distance_matrix="braycurtis"):
        for i in range(t):
            self.trees.append(omri_tree(X, 0, l, psi, distance_matrix, 0))
            self.number_of_trees += 1


'''
    inputs: X=input data, OTUs=taxa, e=current tree depth, l=depth limit, psi=sub sampling size, distance_matrix = chosen function to calcukate distance matrix
    function:
    output: an omriTree represented by it's root (treeNode type)
'''
def omri_tree(X, e, l, psi, distance_matrix, random_state=0):
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
        print("contains null:", renormalized_X.isnull().values.any())
        print("renormalized_X")
        print(renormalized_X)

        renormalized_X.to_csv("/specific/elhanan/PROJECTS/ANOMALY_DETECTION_OP/mat4beta.txt", index=False)
        distance_mat = beta_diversity(distance_matrix, renormalized_X.T, renormalized_X.columns)
        print(distance_mat)
        # distance_mat = renormalized_X

        #print(distance_mat_std.isnull())
        pc1 = pcoa(distance_mat, number_of_dimensions=1)
        print(pc1)
        # pc1 = pcoa(distance_mat, number_of_dimensions=1).samples[['PC1']] ####????? extract 1st pcoa

        # if (distance_mat.sum(axis=0) <= 0).any() or distance_mat.sum(axis=1).any() <= 0:
        #     print(distance_mat)

        # pca = PCA(n_components=1)
        # pc1 = pca.fit_transform(distance_mat.transpose())

        # print("explained variance ratio", pca.explained_variance_ratio_)
        # print("var", pca.explained_variance_)

        #pca_df = pd.DataFrame(data=pc1, columns=['pc1'])

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


def kde(A_depths, B_depths, title, file_name):
    if type(A_depths[0]) == list:
        all_A = []
        for lst in A_depths:
            for m in lst:
                all_A.append(m)
        all_B = []
        for lst in B_depths:
            for m in lst:
                all_B.append(m)
    all_A = A_depths
    all_B = B_depths
    sns.kdeplot(data=all_A, label='outliers')
    sns.kdeplot(data=all_B, label='normals')
    plt.title(title)
    plt.xlabel("depths")
    plt.legend()
    plt.savefig(file_name)
    plt.show()


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
    data = data[data.mean(axis=1) > 0.0]
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


def anomalies_against_normals(anomalies_data, normals_data, t, psi, outliers_percentage, dir_name, times):

    anomalies_data = anomalies_data.loc[:, anomalies_data.sum() > 0] # filter samples with no data
    normals_data = normals_data.loc[:, normals_data.sum() > 0]
    normals_samples = normals_data.columns
    # data = data.T.drop_duplicates().T

    outliers_count = int(outliers_percentage * len(normals_samples) / (100 - outliers_percentage))
    l = int(math.log(len(normals_samples)+outliers_count,2))

    A_depths, B_depths, compare_depths, auc_lst, auc_IF, p, auc_p_r, auc_IF_p_r = [], [], [], [], [], [], [], []

    for i in range(times):
        anomalies_data = anomalies_data.sample(n=outliers_count, axis=1)
        anomalies_samples = anomalies_data.columns
        # data = pd.concat([normals_data, anomalies_data], axis=1) #.reset_index(inplace=True)
        data = concat_data_frames([normals_data, anomalies_data], 1)
        # sick_and_healthy_data = sick_and_healthy_data.iloc[:, 1:]
        data = relative_abundence(data)

        OF = Forest(0,[])
        OF.fit(data, psi, l,t)

        # Test
        now_A_depths, now_B_depths = test_groups_by_depths(anomalies_samples,normals_data, OF)
        A_depths.append(now_A_depths)
        B_depths.append(now_B_depths)
        cd = (sum(now_A_depths) / len(now_A_depths)) < (sum(now_B_depths) / len(now_B_depths))
        compare_depths.append(cd)
        U1, now_p = mannwhitneyu(now_A_depths, now_B_depths, alternative="less")
        p.append(now_p)
        y_true = len(normals_samples) * [1] + len(anomalies_samples) * [0]
        auc_lst.append(roc_auc_score(y_true, now_B_depths + now_A_depths))
        probs = now_B_depths + now_A_depths
        # probs = [x / math.ceil(l) for x in probs]
        precision, recall, thresholds = precision_recall_curve(y_true, probs)
        auc_p_r.append(auc(recall, precision))

        # Isolation Forest test
        model = IsolationForest(n_estimators=t)
        model.fit(data.transpose())
        scores = model.decision_function(data.transpose())  # higher score means more normal
        auc_IF.append(roc_auc_score(y_true, scores))
        precision_IF, recall_IF, thresholds_IF = precision_recall_curve(y_true, scores)
        auc_IF_p_r.append(auc(recall_IF, precision_IF))

    create_dir(dir_name)
    write_list_to_file(dir_name + "/anomalies depths", A_depths)
    write_list_to_file(dir_name + "/normal depths", B_depths)
    write_list_to_file(dir_name + "/compare depths", compare_depths)
    write_list_to_file(dir_name + "/p values", p)
    write_list_to_file(dir_name + "/auc", auc_lst)
    write_list_to_file(dir_name + "/auc IF", auc_IF)
    write_list_to_file(dir_name + "/auc precision recall", auc_p_r)
    write_list_to_file(dir_name + "/auc IF precision recall", auc_IF_p_r)

    if times > 1:
        A_depths = [item for sublist in A_depths for item in sublist]
        B_depths = [item for sublist in B_depths for item in sublist]
    else:
        A_depths = A_depths[0]
        B_depths = B_depths[0]
    kde(A_depths,B_depths, dir_name + " - depths kde", dir_name + "/kde")

    return A_depths, B_depths, compare_depths, p, auc_lst, auc_IF, auc_p_r, auc_IF_p_r

# df.to_csv

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


schubert_data = load_dataset("Diseases/schubert/RDP/cdi_schubert.otu_table.100.denovo.rdp_assigned")
# print(schubert_data)
schubert_metadata = pd.read_csv("Diseases/schubert/cdi_schubert.metadata.txt",sep='\t', encoding= 'unicode_escape')
schubert_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)
schubert_metadata.rename(columns={"sample_id": "#SampleID"}, inplace=True)
schubert_metadata = schubert_metadata[schubert_metadata["DiseaseState"] != "nonCDI"]
schubert_metadata = schubert_metadata[schubert_metadata["#SampleID"] != 'DA00939']
healthy_schubert_metadata = schubert_metadata[schubert_metadata["DiseaseState"] == "H"]
healthy_schubert_data = schubert_data[healthy_schubert_metadata["#SampleID"]]

vincent_data = load_dataset("Diseases/vincent/RDP/cdi_vincent_v3v5.otu_table.100.denovo.rdp_assigned")
# print(vincent_data)
vincent_metadata = load_dataset("Diseases/vincent/cdi_vincent_v3v5.metadata.txt")
vincent_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)
vincent_metadata.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)
healthy_vincent_metadata = vincent_metadata[vincent_metadata["DiseaseState"] == "H"]
healthy_vincent_data = vincent_data[healthy_vincent_metadata["#SampleID"]]



# print(pd.Series(list(set(vincent_data.iloc[:, 0]).intersection(set(schubert_data.iloc[:, 0])))))
# print(healthy_vincent_data)


t = 10
psi = 2000
outliers_percentage = 5
times = 50
dir_name = str(times) + " times bacth effect- healthy schubert and vincent " + str(t) + " trees " + str(outliers_percentage) + "% of outliers " + str(psi) + " sub sample"


anomalies_against_normals(healthy_vincent_data, healthy_schubert_data, t, psi, outliers_percentage, dir_name, times)


def box_plot_hue(auc, auc_IF, title):
    df = pd.DataFrame()
    auc_df = pd.DataFrame()
    auc_df["model"] = ["Omri Tree" for j in range(len(auc))]
    auc_df["auc"] = auc
    auc_df["y"] = "y"
    df = pd.concat([df, auc_df], ignore_index=True)
    auc_IF_df = pd.DataFrame()
    auc_IF_df["model"] = ["Isolation Forest" for j in range(len(auc_IF))]
    auc_IF_df["auc"] = auc_IF
    auc_IF_df["y"] = "y"
    df = pd.concat([df, auc_IF_df], ignore_index=True)

    print(df)
    sns.boxplot(data=df, x="auc",y="y", hue="model").set(title=title)
    # plt.axvline(x=0.95, color='g')
    plt.xlim(0.9,1)
    plt.tight_layout()
    plt.savefig(title)
    plt.show()



# auc_IF_pre_rec = [0.9696944052135748, 0.9674607637610482, 0.9622278986155608, 0.9514865880662584, 0.9680248086737087, 0.9441970930707154, 0.9585491163575185, 0.9519511227669526, 0.9627575403717404, 0.976114775203089, 0.9533271779851913, 0.9395491585945356, 0.9250287227911411, 0.9510106091420222, 0.9633815940345751, 0.9833776139007698, 0.9397271864881851, 0.9758346361761985, 0.9657846618597272, 0.9504452645562891, 0.9412258872990222, 0.9461337173794842, 0.9384337781654057, 0.9667769101406456, 0.9294829914661473, 0.9649421426648626, 0.9587297807263643, 0.9356473733299392, 0.9756004679020635, 0.9365340583822299, 0.9339523511815424, 0.9745465695296507, 0.9611601612534955, 0.9446620499984197, 0.9651004273664223, 0.9404927039395821, 0.9650130046027262, 0.9621462910746436, 0.9652761780859248, 0.9604141958962176, 0.9488825158596407, 0.9469509575679356, 0.9620043034206598, 0.9550377804840569, 0.9660451977110975, 0.9524459361981363, 0.9317507651318204, 0.9653077635491293, 0.9485281882887024, 0.9652077166683417]
# auc_pre_recall = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# box_plot_hue(auc_pre_recall, auc_IF_pre_rec, "auc precision-recall - " + dir_name)