import torch
from torchvision import models, transforms
from torchvision.utils import save_image
from facenet_pytorch import MTCNN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from torchvision.models import ResNet18_Weights
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import shutil
import umap
from sklearn.preprocessing import StandardScaler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MTCNN for face detection
mtcnn = MTCNN(keep_all=False, device=device, image_size=128, margin=30)

# Transform pipeline
transform = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_data(img_dir, mtcnn, transform):
    processed_images = []
    for img_file in tqdm(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img_cropped = mtcnn(img)
        if img_cropped is not None:
            if not isinstance(img_cropped, torch.Tensor):
                img_cropped = transform(img_cropped)
            processed_images.append(img_cropped)
    return torch.stack(processed_images)

def preprocess_data_without_mtcnn(img_dir, transform, target_size=(192, 192)):
    processed_images = []
    for img_file in tqdm(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize(target_size)  # 调整图像尺寸
        img_tensor = transform(img_resized)
        processed_images.append(img_tensor)
    return torch.stack(processed_images)


def extract_features(processed_images, model):
    features = []
    with torch.no_grad():
        for img in tqdm(processed_images):
            feature = model(img.unsqueeze(0).to(device))
            features.append(feature.cpu().numpy().squeeze())
    return np.array(features)

# Load a pre-trained ResNet model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Identity()  # Removing the last fully connected layer
model = model.to(device)


def dimension_reduction_per_cluster(features, labels, n_components_pca=3, n_components_tsne=3):
    reduced_features = np.empty((0, n_components_tsne))
    new_labels = []

    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue  # 跳过噪声聚类

        # 获取当前聚类的特征
        cluster_features = features[labels == cluster_id]

        # 对当前聚类进行降维
        pca = PCA(n_components=n_components_pca)
        tsne = TSNE(n_components=n_components_tsne, perplexity=3)
        # reduced_cluster_features = tsne.fit_transform(pca.fit_transform(cluster_features))
        # reduced_cluster_features = tsne.fit_transform(cluster_features)
        reduced_cluster_features = pca.fit_transform(cluster_features)
        # 合并结果
        reduced_features = np.vstack((reduced_features, reduced_cluster_features))
        new_labels.extend([cluster_id] * len(reduced_cluster_features))

    return reduced_features, np.array(new_labels)

def umap_dimension_reduction(features, n_components=2, n_neighbors=15, min_dist=0.1):
    # 标准化特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 应用UMAP
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)
    reduced_features = reducer.fit_transform(scaled_features)
    return reduced_features

def perform_clustering(features, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(features)
    return labels

def perform_hdbscan_clustering(features, min_cluster_size=5, min_samples=None):
    hdbscan_cluster = HDBSCAN(min_cluster_size=min_cluster_size, 
                              min_samples=min_samples)
    labels = hdbscan_cluster.fit_predict(features)

    # 统计每个聚类的样本数
    unique_labels = np.unique(labels)
    print(f"总共聚类数量（包括噪声）: {len(unique_labels)}")
    for label in unique_labels:
        if label == -1:
            print(f"噪声样本数: {np.sum(labels == label)}")
        else:
            print(f"聚类 {label} 的样本数: {np.sum(labels == label)}")

    return labels


def visualize_clusters_3d(data, labels):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', depthshade=True)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.savefig('T3_3d.png')
    plt.show()

def visualize_clusters(data, labels):
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.savefig('T3.png')
    plt.show()

def select_samples_from_clusters(labels, img_dir, samples_per_cluster=10, output_dir='clustered_samples'):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历标签中的每个聚类
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue  # 如果存在噪声聚类，则跳过
        
        # 为当前聚类创建一个目录
        cluster_dir = os.path.join(output_dir, f'cluster_{cluster_id}')
        os.makedirs(cluster_dir, exist_ok=True)

        # 获取当前聚类中所有样本的索引
        cluster_indices = np.where(labels == cluster_id)[0]

        # 随机选择一部分样本
        selected_indices = np.random.choice(cluster_indices, samples_per_cluster, replace=False)

        # 将选中的样本复制到聚类目录
        for idx in selected_indices:
            # 处理过的图像文件名
            img_file = os.listdir(img_dir)[idx]

            # 源图像路径
            src_img_path = os.path.join(img_dir, img_file)

            # 目标图像路径
            dst_img_path = os.path.join(cluster_dir, img_file)

            # 将图像复制到目标目录
            shutil.copy(src_img_path, dst_img_path)

    print(f"选取的样本已保存到 {output_dir}")


# Replace with your dataset directory
img_dir = 'genki4k\\files'

# Data Preprocessing
#processed_images = preprocess_data(img_dir, mtcnn, transform)

processed_images = preprocess_data_without_mtcnn(img_dir, transform)

# Feature Extraction
features = extract_features(processed_images, model)


# Clustering
labels = perform_clustering(features)
# labels = perform_hdbscan_clustering(features)

# Dimension Reduction per Cluster
# reduced_data, new_labels = dimension_reduction_per_cluster(features, labels)

# 使用UMAP进行降维
reduced_data = umap_dimension_reduction(features, n_components=3)

# Visualization
#visualize_clusters(reduced_data, new_labels)
# visualize_clusters_3d(reduced_data, new_labels)
visualize_clusters_3d(reduced_data, labels)

# Select samples from clusters
# select_samples_from_clusters(new_labels, img_dir, 15, output_dir='clustered_samples')
select_samples_from_clusters(labels, img_dir, 15, output_dir='clustered_samples')
