from torch.utils.data import Dataset
import torch
import pickle
import os
import numpy as np 

class VolleyballB3_stage2(Dataset):
    def __init__(self,pickle_file, 
                 videos_root, video_list,feuture_root, encoder, transform=None):
        
        self.videos_root = videos_root
        self.transform = transform
        self.encoder = encoder
        self.feuture_root=feuture_root

        with open(pickle_file, 'rb') as f:
            self.videos_annot = pickle.load(f)

        self.samples = []
        for video_dir in video_list:
            if video_dir not in self.videos_annot:
                continue
            clips = self.videos_annot[video_dir]
            for clip_dir, clip_data in clips.items():
                
                feuture_path=os.path.join(feuture_root,video_dir,f"{clip_dir}.npz")
                data=np.load(feuture_path)

                features = data['features']   # (12, 2048)
                pooled_feature = np.max(features, axis=0)

                label_str = clip_data['category']
                label_int = self.encoder.encode(label_str)
                self.samples.append((pooled_feature, label_int))

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        pooled_feature, label = self.samples[idx]
        if self.transform:
            pooled_feature = self.transform(pooled_feature)
            
        return pooled_feature, label












