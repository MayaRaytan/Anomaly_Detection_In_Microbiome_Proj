import pandas as pd

def load_dataset(path):
    data = pd.read_csv(path,sep='\t')
    return data




# zeller_data = load_dataset("Diseases/zeller/RDP/crc_zeller.otu_table.100.denovo.rdp_assigned")
# zeller_metadata = pd.read_csv("Diseases/zeller/crc_zeller.metadata.txt", sep='\t', encoding= 'unicode_escape')
# zeller_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)
# zeller_metadata.rename(columns={"Subject ID": "#SampleID"}, inplace=True)


# zhao_data = load_dataset("C:/Maya/CS/AnomalyDetectionMicrobiome/Diseases/zhao/RDP/crc_zhao.otu_table.100.denovo.rdp_assigned")
# zhao_metadata = pd.read_csv("C:/Maya/CS/AnomalyDetectionMicrobiome/Diseases/zhao/crc_zhao.metadata.txt", sep='\t', encoding= 'unicode_escape')
# zhao_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)

# baxter_data = load_dataset("Diseases/baxter/RDP/crc_baxter.otu_table.100.denovo.rdp_assigned")
# baxter_metadata = load_dataset("Diseases/baxter/crc_baxter.metadata.txt")
# baxter_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)
# baxter_metadata.rename(columns={"Sample_Name_s": "#SampleID"}, inplace=True)
# baxter_metadata = baxter_metadata[baxter_metadata["DiseaseState"] != "nonCRC"]
# baxter_metadata["#SampleID"] = baxter_metadata["#SampleID"].astype("string")

# zackular_data = load_dataset("C:/Maya/CS/AnomalyDetectionMicrobiome/Diseases/zackular/RDP/crc_zackular.otu_table.100.denovo.rdp_assigned")
# zackular_metadata = pd.read_csv("C:/Maya/CS/AnomalyDetectionMicrobiome/Diseases/zackular/crc_zackular.metadata.txt", sep='\t', encoding= 'unicode_escape')
# zackular_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)
# zackular_metadata.rename(columns={"sample_id": "#SampleID"}, inplace=True)
# zackular_metadata = zackular_metadata[zackular_metadata["DiseaseState"] != "nonCRC"]
#
# xiang_data = load_dataset("C:/Maya/CS/AnomalyDetectionMicrobiome/Diseases/xiang/RDP/crc_xiang.otu_table.100.denovo.rdp_assigned")
# xiang_metadata = pd.read_csv("C:/Maya/CS/AnomalyDetectionMicrobiome/Diseases/xiang/crc_xiang.metadata.txt", sep='\t', encoding= 'unicode_escape')
# xiang_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)


schubert_data = load_dataset("C:/Maya/CS/AnomalyDetectionMicrobiome/Diseases/schubert/RDP/cdi_schubert.otu_table.100.denovo.rdp_assigned")
schubert_metadata = pd.read_csv("C:/Maya/CS/AnomalyDetectionMicrobiome/Diseases/schubert/cdi_schubert.metadata.txt",sep='\t', encoding= 'unicode_escape')
schubert_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)
schubert_metadata.rename(columns={"sample_id": "#SampleID"}, inplace=True)
schubert_metadata = schubert_metadata[schubert_metadata["DiseaseState"] != "nonCDI"]
schubert_metadata = schubert_metadata[schubert_metadata["#SampleID"] != 'DA00939']


vincent_data = load_dataset("C:/Maya/CS/AnomalyDetectionMicrobiome/Diseases/vincent/RDP/cdi_vincent_v3v5.otu_table.100.denovo.rdp_assigned")
vincent_metadata = load_dataset("C:/Maya/CS/AnomalyDetectionMicrobiome/Diseases/vincent/cdi_vincent_v3v5.metadata.txt")
vincent_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)
vincent_metadata.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)


youndster_data = load_dataset("C:\Maya\CS\AnomalyDetectionMicrobiome\Diseases\youngster\RDP\cdi_youngster.otu_table.100.denovo.rdp_assigned")
youngster_metadata = load_dataset("C:\Maya\CS\AnomalyDetectionMicrobiome\Diseases\youngster\cdi_youngster.metadata.txt")
youndster_data.rename(columns={"Unnamed: 0": "#SampleID"}, inplace=True)
youngster_metadata.rename(columns={"#Sample_id": "#SampleID"}, inplace=True)
youngster_metadata = youngster_metadata[youngster_metadata["DiseaseState"] != "postFMT_CDI"].reset_index(drop=True)
youngster_data = youndster_data[youngster_metadata["#SampleID"]].reset_index(drop=True)


common_features = pd.Series(list(set(schubert_data.iloc[:, 0]).intersection(set(vincent_data.iloc[:, 0]))))
print(common_features)
# print(vincent_data.iloc[:, 1:])
old_sum = vincent_data.iloc[:, 1:].sum()
# print(old_sum)
vincent_data = vincent_data[vincent_data["#SampleID"].isin(common_features)]
# print(vincent_data)
new_sum = vincent_data.iloc[:, 1:].sum()
print((new_sum/old_sum).mean())

