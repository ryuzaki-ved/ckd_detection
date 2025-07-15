# 🩺 Chronic Kidney Disease Prediction using Neural Networks

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This project leverages the power of **Neural Networks** to predict Chronic Kidney Disease (CKD) based on a comprehensive set of health records. Utilizing the UCI Chronic Kidney Disease dataset, this solution provides a robust classification model, complete with detailed data preprocessing, exploratory data analysis, and model interpretability.

> **🔬 Medical AI at its finest** - Helping healthcare professionals make informed decisions through machine learning

## ✨ Key Features

### 🔧 Data Processing & Engineering
- **Smart Missing Value Handling**: Robust imputation using mean/median for numerical and mode for categorical data
- **Data Cleaning**: Automated detection and correction of inconsistent data entries
- **Feature Engineering**: Label encoding, Min-Max scaling, and PCA dimensionality reduction
- **Class Balance**: RandomOverSampler to handle imbalanced datasets

### 📊 Comprehensive Analysis
- **Exploratory Data Analysis**: Interactive visualizations including pairplots, distribution plots, and boxplots
- **Outlier Detection**: Advanced statistical methods to identify and handle outliers
- **Correlation Analysis**: Heatmaps to understand feature relationships

### 🧠 Advanced Neural Network
- **Sequential Keras Model**: Optimized architecture with Dense layers and Dropout
- **Hyperparameter Tuning**: Fine-tuned for optimal performance
- **Early Stopping**: Prevents overfitting during training

### 📈 Model Evaluation & Interpretability
- **Multi-metric Evaluation**: Accuracy, F1-Score, AUC, Precision-Recall curves
- **Confusion Matrix**: Detailed classification performance analysis
- **SHAP Integration**: Explainable AI for transparent decision-making
- **Feature Importance**: Understanding which health indicators matter most

## 📊 Dataset Information

| Attribute | Description | Type |
|-----------|-------------|------|
| **Dataset Source** | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease) | - |
| **Instances** | 400 patients | - |
| **Features** | 24 health indicators | Mixed |
| **Target** | CKD vs Non-CKD | Binary |

### 🏥 Health Indicators Include:
- Age, Blood Pressure, Specific Gravity
- Albumin, Sugar, Red Blood Cells
- Hemoglobin, Packed Cell Volume
- White Blood Cell Count, and more...

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.7+
pip (Python package manager)
```

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd chronic-kidney-disease-prediction

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow shap opencv-python
```

### 🏃‍♀️ Running the Model
```bash
# Run the main prediction script
python code1.py

# For step-by-step analysis
python converted_code.py
```

## 📁 Project Structure
```
📦 chronic-kidney-disease-prediction/
├── 📄 README.md                    # You are here!
├── 📄 LICENSE                      # MIT License
├── 🐍 code1.py                     # Main neural network implementation
├── 🐍 converted_code.py            # Step-by-step analysis
├── 🐍 digital_tissue_mapping.py    # Additional tissue analysis
├── 📊 kidney_disease.csv           # Original dataset
├── 📊 Final_pre_processing_data.csv # Processed dataset
├── 🤖 kidney_disease_model.h5      # Trained model
├── 📝 step_by_step.txt             # Detailed methodology
├── 🔗 helpful_links.txt            # Additional resources
└── 📁 output_images/               # Generated visualizations
```

## 📈 Model Performance

Our Neural Network achieves impressive results:

- **🎯 High Accuracy**: Optimized for medical diagnosis standards
- **⚖️ Balanced Performance**: Handles both CKD and Non-CKD cases effectively
- **🔍 Interpretable**: SHAP values explain each prediction
- **📊 Comprehensive Metrics**: F1-Score, Precision, Recall, AUC

> Run the model to see exact performance metrics!

## 🔬 Advanced Features

### 🧪 SHAP Explainability
```python
# Generate SHAP explanations
explainer = shap.KernelExplainer(model.predict, x_train)
shap_values = explainer.shap_values(x_test[:100])
shap.summary_plot(shap_values, x_test[:100])
```

### 📊 Visualization Gallery
The project generates various visualizations:
- 📈 Training history plots
- 🎯 ROC curves
- 📊 Precision-Recall curves
- 🔥 Confusion matrices
- 🌟 SHAP feature importance plots

## 🛠️ Usage Examples

### Basic Prediction
```python
# Load and preprocess data
df = pd.read_csv("kidney_disease.csv")
# ... preprocessing steps ...

# Train model
model = create_neural_network()
history = model.fit(x_train, y_train, epochs=20)

# Make predictions
predictions = model.predict(x_test)
```

### Model Interpretation
```python
# Explain individual predictions
explain_prediction(model, explainer, sample_index=5)
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **🍴 Fork** the repository
2. **🌿 Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **💾 Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **📤 Push** to the branch (`git push origin feature/AmazingFeature`)
5. **🔄 Open** a Pull Request

### 💡 Ideas for Contributions
- Additional ML algorithms comparison
- Web interface for predictions
- Mobile app integration
- Enhanced visualizations
- Performance optimizations

## 📚 Documentation

- **📖 Step-by-Step Guide**: See `step_by_step.txt` for detailed methodology
- **🔗 Helpful Resources**: Check `helpful_links.txt` for additional materials
- **📊 Research Papers**: Explore the `research_papers/` directory

## 🏆 Acknowledgements

- **🎓 UCI Machine Learning Repository** for the comprehensive dataset
- **🧠 TensorFlow Team** for the amazing deep learning framework
- **📊 SHAP Contributors** for explainable AI capabilities
- **👥 Open Source Community** for continuous inspiration

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 📞 Support & Contact

- **🐛 Issues**: [GitHub Issues](https://github.com/your-username/your-repo/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/your-username/your-repo/discussions)
- **📧 Email**: your-email@example.com

## 🌟 Star History

If this project helped you, please consider giving it a ⭐!

---

<div align="center">

**Made with ❤️ for better healthcare through AI**

[⬆ Back to Top](#-chronic-kidney-disease-prediction-using-neural-networks)

</div>
