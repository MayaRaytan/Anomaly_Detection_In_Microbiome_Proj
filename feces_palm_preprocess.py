import sys

import pandas as pd

from OMRI import load_dataset
from OMRI import write_list_to_file

# outliers_percentage_sample = sys.argv[1]
# palm_part_of_sample = sys.argv[2]

outliers_percentage_sample = 10
palm_part_of_sample = 0.2

palm_F_df = load_dataset("Moving_Pictures/F4_R_palm_L6.txt")
feces_M_df = load_dataset("Moving_Pictures/M3_feces_L6.txt")

# old_sum = palm_F_df.iloc[:, 1:].sum()

# all 330 feces features are in 1200 palm
features_intersection = pd.Series(list(set(feces_M_df["#OTU ID"]).intersection(set(palm_F_df["#OTU ID"]))))
# new_sum = palm_F_df[palm_F_df["#OTU ID"].isin(intersection)].iloc[:, 1:].sum()
# print((new_sum/old_sum).mean())

# print(pd.Series(list(set(feces_M_df["#OTU ID"]).intersection(set(intersection)))))

palm_F_df.set_index(["#OTU ID"], inplace=True)
feces_M_df.set_index(["#OTU ID"], inplace=True)

# result = palm_F_df["#OTU ID"].equals(other=feces_M_df["#OTU ID"])


outliers_count = int((outliers_percentage_sample * len(feces_M_df.columns)) / (100 - outliers_percentage_sample))

samples_intersection = pd.Series(list(set(feces_M_df.columns).intersection(set(palm_F_df.columns))))
# print(samples_intersection)
palm_F_df = palm_F_df[samples_intersection]
palm_F_df = palm_F_df.sample(n=outliers_count, axis=1)

feces80_palm20 = feces_M_df[palm_F_df.columns]
feces80_palm20 = 0.8 * feces80_palm20
# df1.sub(df2, fill_value=0).reindex(df1.index)
# feces_M_df[palm_F_df.columns] = 0.8 * feces_M_df[palm_F_df.columns]
# print(feces_M_df)
palm_F_df = 0.2 * palm_F_df
feces80_palm20 = feces80_palm20.add(palm_F_df, fill_value=0) # adding many palm features
feces80_palm20.reset_index(inplace=True)
# feces80_palm20 = feces80_palm20[feces80_palm20["#OTU ID"].isin(features_intersection)] # leave only feces features

feces_intersection = pd.Series(list(set(feces_M_df.columns).intersection(set(feces80_palm20.columns))))
# print(feces_M_df.columns)
# print(feces80_palm20.columns)
a = feces_M_df.columns.difference(feces_intersection)
feces_M_df = feces_M_df[a]
# compare = (feces_M_df.columns.to_series()).compare(feces_intersection)
# feces_M_df = feces_M_df[feces_M_df.columns.isin(compare)]

feces_M_df.reset_index(inplace=True)
feces80_palm20 = feces80_palm20.iloc[:, 1:]
feces_M_df = feces_M_df.iloc[:, 1:]


feces_M_df.to_csv("Moving_Pictures/only_feces.txt", sep='\t')
feces80_palm20.to_csv("Moving_Pictures/80%feces_20%palm_5&outliers.txt", sep='\t')
# write_list_to_file("Moving_Pictures/80%feces_20%palm_5&outliers_samples.txt", list(palm_F_df.columns))


# for col in palm_F_df:
#     feces_M_df[col] = (1 - palm_part_of_sample) * feces_M_df[col] + palm_part_of_sample * palm_F_df[col]

