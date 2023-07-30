import pandas as pd
import os
df = pd.read_csv('./train_df.csv')
df['file_no'] = df['Participant_ID'].apply(lambda x : x.split('_')[0])
# for files in os.listdir('./Train_images'):


file_names = os.listdir('./Train_images')
final_df = pd.DataFrame({
    "file_name" : file_names
})

final_df['file_no'] = final_df['file_name'].apply(lambda x : x[:3])

final_df['file_no'] =pd.to_numeric(final_df['file_no'])
df['file_no'] = pd.to_numeric(df['file_no'])
df2 = df.join(final_df.set_index('file_no'), on='file_no', how='inner')
df2.drop(['Participant_ID', 'file_no'],axis=1, inplace=True)

df2.to_csv("data_set_for_training.csv", index=False)

        
