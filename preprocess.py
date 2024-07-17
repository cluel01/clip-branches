import pandas as pd
import os

image_list = []
with open('shutterstock.csv', 'r') as file:
    # Read each line in a loop
    for n,line in enumerate(file):
        # Do something with the line
        l = line.strip()
        try:
            #l = l.replace("\"","")
            image_url,text = l.split("\t")[:2]
        except:
            continue
        image_list.append([image_url,text])

df = pd.DataFrame(image_list,columns=["image_url","caption"])

image_dataset = []

for idx,row in df.iterrows():
    filename = os.path.join("images",f"image_{idx}.jpg")
    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        image_dataset.append([filename,row.caption])
        
df_clip = pd.DataFrame(image_dataset,columns=["filename","caption"])
df_clip.to_csv("data/shutterstock_dataset.csv",index=False)