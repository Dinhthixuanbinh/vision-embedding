import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
import faiss
from tqdm import tqdm

class ImageRetrieval:
    def __init__(self, model_name="hf-hub:apple/DFN5B-CLIP-ViT-H-14-384", device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model and preprocessor
        self.model, self.preprocess = create_model_from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = get_tokenizer('ViT-H-14')
        
        # Initialize FAISS index
        self.index = None
        self.image_paths = []

    def preprocess_images(self, image_paths):
        images = [self.preprocess(Image.open(path).convert("RGB")).unsqueeze(0) for path in image_paths]
        return torch.cat(images, dim=0).to(self.device)

    def extract_features(self, image_paths, batch_size=4):
        all_features = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i+batch_size]
            images = self.preprocess_images(batch_paths)
            with torch.no_grad(), torch.cuda.amp.autocast():
                features = self.model.encode_image(images)
                features = F.normalize(features, dim=-1)
            all_features.append(features.cpu().numpy())
        
        return np.vstack(all_features)

    def build_index(self, image_directory):
        image_paths = []
        for root, _, files in os.walk(image_directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(image_paths)} images. Building index...")
        features = self.extract_features(image_paths)
        
        self.index = faiss.IndexFlatIP(features.shape[1])
        self.index.add(features)
        self.image_paths = image_paths
        print("Index built successfully.")

    def search(self, query_image_path, k=5):
        query_feature = self.extract_features([query_image_path])
        D, I = self.index.search(query_feature, k)
        return [(self.image_paths[i], D[0][j]) for j, i in enumerate(I[0])]

    def search_with_text(self, text_query, k=5):
        text = self.tokenizer([text_query], context_length=self.model.context_length).to(self.device)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.model.encode_text(text)
            text_features = F.normalize(text_features, dim=-1)
        
        D, I = self.index.search(text_features.cpu().numpy(), k)
        return [(self.image_paths[i], D[0][j]) for j, i in enumerate(I[0])]

# Usage example
if __name__ == "__main__":
    # Initialize the image retrieval system
    retriever = ImageRetrieval()
    
    # Specify the directory containing your images
    image_directory = '/kaggle/input/cutframe-aic24-l1314/Keyframes'
    
    retriever.build_index(image_directory)
    
    # Image-based search
    query_image_path = '/kaggle/input/cutframe-aic24-l1314/Keyframes/test/0001_7395/0001.jpg'
    print("Searching for similar images...")
    results = retriever.search(query_image_path, k=5)
    
    print("Image-based search results:")
    for i, (path, score) in enumerate(results, 1):
        print(f"{i}. Image: {path}")
        print(f"   Similarity Score: {score:.4f}")
        print(f"   Image ID: {os.path.basename(os.path.dirname(path))}")
        print()
    
    # Text-based search
    text_query = "a person walking on the street"
    print(f"Searching for images matching the text: '{text_query}'")
    text_results = retriever.search_with_text(text_query, k=5)
    
    print("Text-based search results:")
    for i, (path, score) in enumerate(text_results, 1):
        print(f"{i}. Image: {path}")
        print(f"   Similarity Score: {score:.4f}")
        print(f"   Image ID: {os.path.basename(os.path.dirname(path))}")
        print()
    
    # Visualization (optional)
    from PIL import Image
    import matplotlib.pyplot as plt
    
    def visualize_results(query, results, is_image_query=True):
        plt.figure(figsize=(20, 4))
        if is_image_query:
            plt.subplot(1, 6, 1)
            plt.imshow(Image.open(query))
            plt.title("Query Image")
            plt.axis('off')
        else:
            plt.figtext(0.5, 0.9, f"Query Text: {query}", ha="center", va="top", fontsize=12)
        
        for i, (path, _) in enumerate(results, 2 if is_image_query else 1):
            plt.subplot(1, 6, i)
            plt.imshow(Image.open(path))
            plt.title(f"Match {i-1 if is_image_query else i}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    visualize_results(query_image_path, results)
    visualize_results(text_query, text_results, is_image_query=False)
