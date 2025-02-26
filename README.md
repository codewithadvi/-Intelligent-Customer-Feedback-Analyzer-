# Intelligent Customer Feedback Analyzer

## **Project Overview**
The **Intelligent Customer Feedback Analyzer** is an AI-powered tool designed to help businesses efficiently process and analyze large volumes of customer feedback. By combining sentiment analysis, topic modeling, and recommendation systems, this project extracts actionable insights from customer reviews. This solution helps improve customer satisfaction, operational efficiency, and decision-making.

The analyzer leverages state-of-the-art technologies such as **DistilBERT** for sentiment analysis, **BERTopic** for topic modeling, and **Sentence Transformers** for recommendations. The pipeline is designed to preprocess text data, analyze sentiment, discover topics, and provide recommendations based on customer feedback.

## **Features**
1. **Sentiment Analysis**:
   - **VADER Sentiment Analyzer**: A lexicon-based approach for analyzing sentiment.
   - **DistilBERT Sentiment Analysis**: A transformer-based approach for fine-grained sentiment classification using a fine-tuned DistilBERT model.
   
2. **Topic Modeling**:
   - Uses **BERTopic** for extracting meaningful topics from customer feedback, with features like:
     - **Vectorization** using **CountVectorizer** for feature extraction.
     - **Dimensionality Reduction** with **UMAP** to reduce the embeddings' dimensionality.
     - **Clustering** with **HDBSCAN** for grouping topics.
     - **Topic Representation** via **c-TF-IDF** for extracting keywords associated with each topic.
     - **Enhancement** of topics using transformer-based embeddings for more accurate and contextual clustering.

3. **Recommendation System**:
   - Generates **similar reviews** based on topics using **Cosine Similarity** and **Sentence Embeddings**.
   - Customizable similarity balancing and topic thresholding to fine-tune results.
   - **Precomputed Embeddings** for faster inference and joblib saving for reusability.

4. **Text Preprocessing & Feature Engineering**:
   - **Preprocessing**: Tokenization, stopword removal, and non-alphabetic character removal.
   - **Visualization**: Word clouds, bigrams, review length analysis, and emoji usage insights.

## **Problem Statement**
The sheer volume of unstructured customer feedback that businesses receive daily makes it difficult to process and extract valuable insights manually. The challenge is compounded by the varied formats and content of feedback. This project aims to automate the process, providing businesses with a scalable solution to analyze feedback efficiently and extract actionable insights, thus improving decision-making, customer service, and overall satisfaction.

## **Model Pipeline**
The analysis is broken down into the following key steps:

### 1. **Text Preprocessing & Sentiment Analysis**
   - **Data Loading**: Customer reviews are loaded and preprocessed for further analysis.
   - **Sentiment Analysis**: 
     - **VADER Sentiment Analyzer** is used for lexicon-based sentiment classification.
     - **DistilBERT** is fine-tuned on the Amazon Reviews dataset for higher accuracy and better sentiment classification.

#### **Advanced Preprocessing**
   - **Emoji Handling**: Emojis are preprocessed by converting them into their respective textual descriptions. This enhances the sentiment analysis by accounting for the emotional tone conveyed by emojis.
   - **Negation Handling**: Negation words (e.g., "not", "never") are specially handled to avoid misinterpretation of sentiment.
   - **Bigrams**: Bigrams (two consecutive words) are extracted to capture commonly used phrases that might convey more meaning than single words.
   - **Word Cloud Generation**: Visualizations of frequent words provide insight into common themes across feedback.
   - **Stopword Removal**: Common, non-informative words (like "the", "is", etc.) are removed to focus on important content.

### 2. **Topic Modeling with BERTopic**
   - **Feature Extraction**: Text is vectorized using **CountVectorizer** to convert reviews into numerical features.
   - **Dimensionality Reduction**: **UMAP** reduces the dimensionality of embeddings to make topic identification easier.
   - **Clustering**: **HDBSCAN** is used for clustering reviews based on topics.
   - **Topic Representation**: **c-TF-IDF** is used to extract important keywords for each topic.
   - **Enhancing Topics**: Transformer-based embeddings enhance the clustering, providing more accurate topic representation.

### 3. **Recommendation System**
   - **Topic Matching**: Reviews are assigned to topics, and **Sentence Transformers** compute embeddings to find similarities between reviews.
   - **Cosine Similarity**: This metric is used to match reviews based on topic keywords.
   - **Customization**: Users can tweak the alpha value for similarity balance and adjust the threshold for topic relevance.
   - **Joblib Saving**: All models are saved as joblib files for easy reuse without the need for retraining.

### 4. **Model Setup & Training**
   - **DistilBERT Fine-Tuning**: The pre-trained **distilbert-base-uncased** model is fine-tuned on the Amazon Reviews dataset.
   - **Metrics**: The model achieves a final **accuracy of 93.75%**, **Precision** of 93.78%, **Recall** of 93.75%, and **F1 Score** of 93.75%.
   - **Training Process**: Hugging Face's **Trainer API** is used to fine-tune the model, with mixed precision for faster training and efficient memory use.

### 5. **Model Evaluation**
   - Sentiment analysis is evaluated on a validation dataset using metrics like **Accuracy**, **Precision**, **Recall**, and **F1 Score**.
   - The model's performance is optimal with the **Jaccard score** of 88.13%, confirming its effectiveness in classifying sentiment accurately.

## **Deployment**
The model is deployed using **Streamlit** for real-time customer feedback analysis. However, the **joblib** files are too large to upload to GitHub, so I've provided **Google Drive links** to access these files directly. You can download the joblib files and use them in your environment.

### **Deployment Issues**:
While the model and sentiment analysis pipeline work perfectly, I encountered an issue with the **requirements.txt** file, which prevents the Streamlit dashboard from running as intended. Due to dependency issues, the dashboard is not currently functional. If time permits, I will resolve these issues and make the Streamlit interface work smoothly.

### **Next Steps**:
1. **Fixing Deployment**: Resolve the requirements.txt and dependency issues to get the Streamlit dashboard working.
2. **Scalability**: Improve the processing time and scalability of the pipeline for large datasets by optimizing memory usage and inference speed.
3. **Model Enhancement**: Continue to refine the sentiment analysis model and BERTopic topic modeling for better accuracy and context.

## **Metrics and Evaluation**

### **Training Results**:
- **Training Loss**: 0.000700 (final epoch)
- **Validation Loss**: 0.537547 (final epoch)
- **Accuracy**: 93.75%
- **Precision**: 93.78%
- **Recall**: 93.75%
- **F1 Score**: 93.75%
- **IoU (Jaccard score)**: 88.13%

### **Sentiment Analysis Metrics**:
- **VADER Sentiment Accuracy**: Evaluated on test data, showcasing the lexicon-based accuracy in quick sentiment classification.
- **DistilBERT Fine-Tuned Accuracy**: 93.75% accuracy achieved, demonstrating the transformer-based model's fine-tuned performance.
- **Precision and Recall**: Both metrics show balanced performance, ensuring the model correctly identifies both positive and negative sentiments.
- **Jaccard Score (IoU)**: The Jaccard score of 88.13% confirms the fine-grained capability of the model to capture sentiment accurately, even in cases with high word overlap.

## **Installation Instructions**
1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd <repository-name>
