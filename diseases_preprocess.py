import pandas as pd

def load_dataset(path):
    data = pd.read_csv(path,sep='\t')
    return data

# change features to genius level only
def genius_level(str):
    i = str[::-1].find(";", 0, len(str))
    i = str[::-1].find(";", i + 1, len(str))
    return str[:len(str) - i]


# baxter_data = load_dataset("Diseases/baxter/RDP/crc_baxter.otu_table.100.denovo.rdp_assigned")
# baxter_metadata = load_dataset("Diseases/baxter/crc_baxter.metadata.txt")
# baxter_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)
# baxter_metadata.rename(columns={"Sample_Name_s": "#SampleID"}, inplace=True)
# baxter_metadata = baxter_metadata[baxter_metadata["DiseaseState"] != "nonCRC"]
# baxter_metadata["#SampleID"] = baxter_metadata["#SampleID"].astype("string")
# healthy_baxter_metadata = baxter_metadata[baxter_metadata["DiseaseState"] == "H"]
# baxter_data["#SampleID"] = baxter_data["#SampleID"].apply(genius_level)

# zeller_data = load_dataset("Diseases/zeller/RDP/crc_zeller.otu_table.100.denovo.rdp_assigned")
# zeller_metadata = pd.read_csv("Diseases/zeller/crc_zeller.metadata.txt", sep='\t', encoding= 'unicode_escape')
# zeller_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)
# zeller_metadata.rename(columns={"Subject ID": "#SampleID"}, inplace=True)
# healthy_zeller_metadata = zeller_metadata[zeller_metadata["DiseaseState"] == "H"]
# zeller_data["#SampleID"] = zeller_data["#SampleID"].apply(genius_level)
#
# zhao_data = load_dataset("C:/Maya/CS/AnomalyDetectionMicrobiome/Diseases/zhao/RDP/crc_zhao.otu_table.100.denovo.rdp_assigned")
# zhao_metadata = pd.read_csv("C:/Maya/CS/AnomalyDetectionMicrobiome/Diseases/zhao/crc_zhao.metadata.txt", sep='\t', encoding= 'unicode_escape')
# zhao_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)
# healthy_zhao_metadata = zhao_metadata[zhao_metadata["DiseaseState"] == "H"]
# zhao_data["#SampleID"] = zhao_data["#SampleID"].apply(genius_level)
#
# old_sum = zhao_data.iloc[:, 1:].sum()
#
# common_features = pd.Series(list(set(zeller_data["#SampleID"]).intersection(set(zhao_data["#SampleID"]))))
# # print(common_features)
#
# zeller_data = zeller_data[zeller_data["#SampleID"].isin(common_features)]
# zhao_data = zhao_data[zhao_data["#SampleID"].isin(common_features)]
#
# zeller_data = zeller_data.groupby(["#SampleID"]).sum().reset_index()
# zhao_data = zhao_data.groupby(['#SampleID']).sum().reset_index()
#
#
#
# zeller_intersection = pd.Series(list(set(healthy_zeller_metadata["#SampleID"]).intersection(set(zeller_data.columns[1:]))))
# healthy_zeller_data = zeller_data[zeller_intersection]
# healthy_zhao_data = zhao_data[healthy_zhao_metadata["#SampleID"]]
#
# new_sum = zhao_data.iloc[:, 1:].sum()
# print((new_sum/old_sum).mean())
#
#
#
# # healthy_zeller_data.to_csv("healthy_zeller_data.txt", sep='\t')
# # healthy_zhao_data.to_csv("healthy_zhao_data.txt", sep='\t')
#
# zh = load_dataset("healthy_zhao_data.txt").iloc[:, 1:]
# print(zh)
# # print(old_sum)
# vincent_data = vincent_data[vincent_data["#SampleID"].isin(common_features)]
# # print(vincent_data)
# new_sum = vincent_data.iloc[:, 1:].sum()
# print((new_sum/old_sum).mean())


# zackular_data = load_dataset("C:/Maya/CS/AnomalyDetectionMicrobiome/Diseases/zackular/RDP/crc_zackular.otu_table.100.denovo.rdp_assigned")
# zackular_metadata = pd.read_csv("C:/Maya/CS/AnomalyDetectionMicrobiome/Diseases/zackular/crc_zackular.metadata.txt", sep='\t', encoding= 'unicode_escape')
# zackular_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)
# zackular_metadata.rename(columns={"sample_id": "#SampleID"}, inplace=True)
# zackular_metadata = zackular_metadata[zackular_metadata["DiseaseState"] != "nonCRC"]
#
# xiang_data = load_dataset("C:/Maya/CS/AnomalyDetectionMicrobiome/Diseases/xiang/RDP/crc_xiang.otu_table.100.denovo.rdp_assigned")
# xiang_metadata = pd.read_csv("C:/Maya/CS/AnomalyDetectionMicrobiome/Diseases/xiang/crc_xiang.metadata.txt", sep='\t', encoding= 'unicode_escape')
# xiang_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)



# good version
schubert_data = load_dataset("Diseases/schubert/RDP/cdi_schubert.otu_table.100.denovo.rdp_assigned")
schubert_metadata = pd.read_csv("Diseases/schubert/cdi_schubert.metadata.txt",sep='\t', encoding= 'unicode_escape')
schubert_metadata.rename(columns={"sample_id": "#SampleID"}, inplace=True)
schubert_metadata = schubert_metadata[schubert_metadata["DiseaseState"] != "nonCDI"]
schubert_metadata = schubert_metadata[schubert_metadata["#SampleID"] != 'DA00939']
healthy_schubert_metadata = schubert_metadata[schubert_metadata["DiseaseState"] == "H"]
schubert_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)
schubert_data["#SampleID"] = schubert_data["#SampleID"].apply(genius_level)

# old_sum = schubert_data[healthy_schubert_metadata["#SampleID"]].sum()


vincent_data = load_dataset("Diseases/vincent/RDP/cdi_vincent_v3v5.otu_table.100.denovo.rdp_assigned")
vincent_metadata = load_dataset("Diseases/vincent/cdi_vincent_v3v5.metadata.txt")
healthy_vincent_metadata = vincent_metadata[vincent_metadata["DiseaseState"] == "H"].reset_index()
healthy_vincent_metadata.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)

vincent_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)
vincent_data["#SampleID"] = vincent_data["#SampleID"].apply(genius_level)


common_features = pd.Series(list(set(vincent_data["#SampleID"]).intersection(set(schubert_data["#SampleID"]))))
# print("")
# print("common features")
# print(common_features)

vincent_data = vincent_data[vincent_data["#SampleID"].isin(common_features)]
schubert_data = schubert_data[schubert_data["#SampleID"].isin(common_features)]

schubert_data = schubert_data.groupby(["#SampleID"]).sum().reset_index()
vincent_data = vincent_data.groupby(['#SampleID']).sum().reset_index()

healthy_schubert_data = schubert_data[healthy_schubert_metadata["#SampleID"]]
healthy_vincent_data = vincent_data[healthy_vincent_metadata["#SampleID"]]

healthy_schubert_data.to_csv("healthy_schubert_data.txt", sep='\t')
healthy_vincent_data.to_csv("healthy_vincent_data.txt", sep='\t')

zh = load_dataset("healthy_schubert_data.txt").iloc[:, 1:]
print(zh)
zh = load_dataset("healthy_vincent_data.txt").iloc[:, 1:]
print(zh)
# new_sum = healthy_schubert_data.sum()
# print((new_sum/old_sum).mean())


#other version
# schubert_data = load_dataset("Diseases/schubert/RDP/cdi_schubert.otu_table.100.denovo.rdp_assigned")
# schubert_metadata = pd.read_csv("Diseases/schubert/cdi_schubert.metadata.txt",sep='\t', encoding= 'unicode_escape')
# schubert_metadata.rename(columns={"sample_id": "#SampleID"}, inplace=True)
# schubert_metadata = schubert_metadata[schubert_metadata["DiseaseState"] != "nonCDI"]
# schubert_metadata = schubert_metadata[schubert_metadata["#SampleID"] != 'DA00939']
# schubert_data.loc[:, 0] = schubert_data.iloc[:, 0].apply(genius_level)
# schubert_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)
# healthy_schubert_metadata = schubert_metadata[schubert_metadata["DiseaseState"] == "H"]
#
# vincent_data = load_dataset("Diseases/vincent/RDP/cdi_vincent_v3v5.otu_table.100.denovo.rdp_assigned")
# vincent_metadata = load_dataset("Diseases/vincent/cdi_vincent_v3v5.metadata.txt")
# healthy_vincent_metadata = vincent_metadata[vincent_metadata["DiseaseState"] == "H"].reset_index()
# healthy_vincent_metadata.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)
#
# vincent_data.loc[:, 0] = vincent_data.iloc[:, 0].apply(genius_level)
# vincent_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)
#
# common_features = pd.Series(list(set(vincent_data["#SampleID"]).intersection(set(schubert_data["#SampleID"]))))
# vincent_data = vincent_data[vincent_data.iloc[:, 0].isin(common_features)]
# schubert_data = schubert_data[schubert_data.iloc[:, 0].isin(common_features)]
#
# schubert_data = schubert_data.groupby(["#SampleID"]).sum().reset_index()
# vincent_data = vincent_data.groupby(['#SampleID']).sum().reset_index()
#
# healthy_schubert_data = schubert_data[healthy_schubert_metadata["#SampleID"]]
# healthy_vincent_data = vincent_data[healthy_vincent_metadata["#SampleID"]]

# common_features = pd.Series(list(set(healthy_vincent_data.iloc[:, 0]).intersection(set(healthy_schubert_data.iloc[:, 0]))))
# healthy_vincent_data = healthy_vincent_data[healthy_vincent_data.iloc[:, 0].isin(common_features)].reset_index()
# healthy_schubert_data = healthy_schubert_data[healthy_schubert_data.iloc[:, 0].isin(common_features)].iloc[:,1:]
# print(healthy_vincent_data)
# # print(vincent_data.iloc[:, 1:])
# old_sum = vincent_data.iloc[:, 1:].sum()
# # print(old_sum)
# vincent_data = vincent_data[vincent_data["#SampleID"].isin(common_features)]
# # print(vincent_data)
# new_sum = vincent_data.iloc[:, 1:].sum()
# print(new_sum/old_sum)




# youndster_data = load_dataset("C:\Maya\CS\AnomalyDetectionMicrobiome\Diseases\youngster\RDP\cdi_youngster.otu_table.100.denovo.rdp_assigned")
# youngster_metadata = load_dataset("C:\Maya\CS\AnomalyDetectionMicrobiome\Diseases\youngster\cdi_youngster.metadata.txt")
# youndster_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)
# youngster_metadata.rename(columns={"#Sample_id": "#SampleID"}, inplace=True)
# youngster_metadata = youngster_metadata[youngster_metadata["DiseaseState"] != "postFMT_CDI"].reset_index(drop=True)
# youngster_data = youndster_data[youngster_metadata["#SampleID"]].reset_index(drop=True)
#
#
# common_features = pd.Series(list(set(schubert_data.iloc[:, 0]).intersection(set(vincent_data.iloc[:, 0]))))
# print(common_features)
# # print(vincent_data.iloc[:, 1:])
# old_sum = vincent_data.iloc[:, 1:].sum()
# # print(old_sum)
# vincent_data = vincent_data[vincent_data["#SampleID"].isin(common_features)]
# # print(vincent_data)
# new_sum = vincent_data.iloc[:, 1:].sum()
# print((new_sum/old_sum).mean())

