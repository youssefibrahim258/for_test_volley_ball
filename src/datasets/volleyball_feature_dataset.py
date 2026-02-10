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

                features = data['features']   # (num_players, 2048)
                max_feat = np.max(features, axis=0)
                mean_feat = np.mean(features, axis=0)
                pooled_feature = np.concatenate([max_feat, mean_feat])

                label_str = clip_data['category']
                label_int = self.encoder.encode(label_str)
                self.samples.append((pooled_feature, label_int))

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        pooled_feature, label = self.samples[idx]
        pooled_feature = torch.from_numpy(pooled_feature).float()
        if self.transform:
            pooled_feature = self.transform(pooled_feature)
        return pooled_feature, label












