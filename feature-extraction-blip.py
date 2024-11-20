import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model and preprocessor
        self.model, self.preprocess = create_model_from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess_images(self, image_paths):
        images = [self.preprocess(Image.open(path).convert("RGB")).unsqueeze(0) for path in image_paths]
        return torch.cat(images, dim=0).to(self.device)

    def extract_features(self, image_directory, save_dir, batch_size=4):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for part in ['train', 'test']:
            part_dir = os.path.join(image_directory, part)
            if not os.path.exists(part_dir):
                continue
            
            save_part_dir = os.path.join(save_dir, part)
            if not os.path.exists(save_part_dir):
                os.makedirs(save_part_dir)

            video_dirs = [d for d in os.listdir(part_dir) if os.path.isdir(os.path.join(part_dir, d))]
            
            for video_dir in tqdm(video_dirs, desc=f"Processing {part} videos"):
                video_path = os.path.join(part_dir, video_dir)
                image_paths = [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                
                if not image_paths:
                    continue

                video_features = []
                for i in range(0, len(image_paths), batch_size):
                    batch_paths = image_paths[i:i+batch_size]
                    images = self.preprocess_images(batch_paths)
                    
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        features = self.model.encode_image(images)
                        features = F.normalize(features, dim=-1)
                    
                    video_features.append(features.cpu().numpy())

                video_features = np.vstack(video_features)
                
                # Save features for this video
                save_path = os.path.join(save_part_dir, f"{video_dir}.npy")
                np.save(save_path, video_features)

        print("Feature extraction and saving completed.")

# Usage example
if __name__ == "__main__":
    extractor = FeatureExtractor()
    
    # Specify the directories
    image_directory = '/kaggle/input/cutframe-aic24-l1314/Keyframes'
    features_directory = '/kaggle/working/Keyframes_features'
    
    # Extract and save features
    extractor.extract_features(image_directory, features_directory, batch_size=4)

    print("Feature extraction process completed.")
    print(f"Features saved in: {features_directory}")

