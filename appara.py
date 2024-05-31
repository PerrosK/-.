import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Load and preprocess the data
def load_data(file):
    data = pd.read_csv(file)
    data.fillna(0, inplace=True)
    return data

def preprocess_data(df):
    numerical_features = df.select_dtypes(include=['number']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    preprocess_pipeline = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )
    
    return preprocess_pipeline, numerical_features, categorical_features

def perform_dimensionality_reduction(df, method):
    preprocess_pipeline, numerical_features, _ = preprocess_data(df)
    
    if method == "PCA":
        dim_reduction = PCA(n_components=2)
    elif method == "t-SNE":
        dim_reduction = TSNE(n_components=2, random_state=42)
    
    pipeline = Pipeline([
        ('preprocessor', preprocess_pipeline),
        ('dim_reduction', dim_reduction)
    ])
    
    X_reduced = pipeline.fit_transform(df)
    
    plot_dimensionality_reduction(X_reduced, method)
    plot_heatmap(df[numerical_features])
    
def plot_dimensionality_reduction(X_reduced, method):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
    plt.xlabel(f'{method} Component 1')
    plt.ylabel(f'{method} Component 2')
    plt.title(f'{method} Visualization')
    st.pyplot(plt)
    
def plot_heatmap(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Heatmap of Numerical Features')
    st.pyplot(plt)

# Function to run classification algorithms and compare them
def classification_algorithms(df, target_column, n_neighbors, max_depth):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    preprocess_pipeline, _, _ = preprocess_data(X)
    
    X_processed = preprocess_pipeline.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

    knn.fit(X_train, y_train)
    dt.fit(X_train, y_train)

    knn_preds = knn.predict(X_test)
    dt_preds = dt.predict(X_test)

    knn_acc = accuracy_score(y_test, knn_preds)
    dt_acc = accuracy_score(y_test, dt_preds)

    st.write(f'k-NN Accuracy: {knn_acc}')
    st.write(f'Decision Tree Accuracy: {dt_acc}')

    st.write("k-NN Confusion Matrix")
    knn_cm = confusion_matrix(y_test, knn_preds)
    knn_disp = ConfusionMatrixDisplay(confusion_matrix=knn_cm)
    knn_disp.plot()
    st.pyplot(plt)

    st.write("Decision Tree Confusion Matrix")
    dt_cm = confusion_matrix(y_test, dt_preds)
    dt_disp = ConfusionMatrixDisplay(confusion_matrix=dt_cm)
    dt_disp.plot()
    st.pyplot(plt)

# Function to run clustering algorithms and evaluate them
def clustering_algorithms(df, n_clusters):
    preprocess_pipeline, _, _ = preprocess_data(df)
    
    X_processed = preprocess_pipeline.fit_transform(df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    agg = AgglomerativeClustering(n_clusters=n_clusters)

    kmeans_labels = kmeans.fit_predict(X_processed)
    agg_labels = agg.fit_predict(X_processed)

    kmeans_silhouette = silhouette_score(X_processed, kmeans_labels)
    agg_silhouette = silhouette_score(X_processed, agg_labels)

    kmeans_db = davies_bouldin_score(X_processed, kmeans_labels)
    agg_db = davies_bouldin_score(X_processed, agg_labels)

    kmeans_ch = calinski_harabasz_score(X_processed, kmeans_labels)
    agg_ch = calinski_harabasz_score(X_processed, agg_labels)

    st.write(f'k-Means Silhouette Score: {kmeans_silhouette}')
    st.write(f'Agglomerative Clustering Silhouette Score: {agg_silhouette}')

    st.write(f'k-Means Davies-Bouldin Score: {kmeans_db}')
    st.write(f'Agglomerative Clustering Davies-Bouldin Score: {agg_db}')

    st.write(f'k-Means Calinski-Harabasz Score: {kmeans_ch}')
    st.write(f'Agglomerative Clustering Calinski-Harabasz Score: {agg_ch}')

    plot_clusters(X_processed, kmeans_labels, "k-Means Clustering")
    plot_clusters(X_processed, agg_labels, "Agglomerative Clustering")

def plot_clusters(X_processed, labels, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_processed[:, 0], X_processed[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    st.pyplot(plt)

# Streamlit interface
st.title('Machine Learning Data Processing Application')

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    data = load_data(uploaded_file)
    tab1, tab2, tab3 = st.tabs(["2D Visualization", "Classifying Algorithm", "Clustering Algorithms"])
    
    with tab1:
        st.subheader('Dimensionality Reduction')
        method = st.selectbox("Select Dimensionality Reduction Method", ["t-SNE", "PCA"])
        if st.button("Run Dimensionality Reduction"):
            perform_dimensionality_reduction(data, method)
    
    with tab2:
        st.subheader('Classification Algorithms')
        target_column = data.columns[-1]  # Automatically select the last column as the target column
        n_neighbors = st.number_input('Enter k for k-NN', min_value=1, max_value=20, value=5)
        max_depth = st.number_input('Enter max depth for Decision Tree', min_value=1, max_value=20, value=5)
        if st.button("Run Classification"):
            classification_algorithms(data, target_column, n_neighbors, max_depth)
    
    with tab3:
        st.subheader('Clustering Algorithms')
        n_clusters = st.number_input('Enter number of clusters', min_value=2, max_value=20, value=3)
        if st.button("Run Clustering"):
            clustering_algorithms(data, n_clusters)
