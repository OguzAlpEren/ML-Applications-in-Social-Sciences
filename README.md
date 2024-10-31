# Projects in Advanced Machine Learning

**Author**: Oguz Alp Eren  
**Institution**: Columbia University, 2023  

## Project Descriptions

### Assignment #1: Write up a report on UN World Happiness Data
**File**: [UN World Happiness Data.ipynb](./Assignment%20%231/Write%20up%20a%20report%20on%20UN%20World%20Happiness%20Data.ipynb)  
Assignment 1 focuses on analyzing the UN World Happiness Dataset, which measures happiness across different countries using factors like GDP per capita, social support, and life expectancy. The report explores correlations, revealing that higher happiness scores are associated with positive indicators in these areas.

**Models Built and Tested**:
1. Random Forest Classifier (n_estimators=250, max_depth=4)
2. GradientBoostingClassifier (n_estimators=45, learning_rate=1.2, max_depth=1)
3. Deep Learning model with relu activation and hidden layers (64, 32, 16, 16)
4. Random Forest with GridSearchCV (cv=10, max_depth=7, n_estimators=110)
5. GradientBoosting with GridSearchCV (cv=10, learning_rate=1.1, max_depth=5, n_estimators=49)
6. Complex Deep Learning model with SGD optimizer, batch_size=20, epochs=350, validation_split=0.25

Findings reveal **GDP per capita** as a strong predictor, while **region** and **generosity** show weaker associations, providing insights for policymakers aiming to improve citizens' well-being.

### Assignment #2: COVID-19 Positive X-Ray Image Data
**File**: [COVID-19 Positive X Ray Image Data.ipynb](./Assignment%20%232/Covid%20Positive%20X%20Ray%20image%20data.ipynb)  
Assignment 2 utilizes a COVID-19 chest X-ray dataset, with the goal of building models to distinguish COVID-19 positive cases from viral pneumonia and normal X-rays. This project aims to support AI-driven diagnostics in medical settings.

**Models Developed**:
1. Custom CNN architecture
2. VGG16 pre-trained model (Transfer Learning)
3. ResNet50V2 Model (Transfer Learning)
4. Improved CNN with L2 Regularization
5. Improved VGG16 (Transfer Learning)
6. Improved ResNet50V2 (Transfer Learning)

Model enhancements included regularization, dropout layers, and transfer learning. Models 5 and 6 performed best for generalization, showing potential in COVID-19 diagnostics.

### Assignment #3: Text Classification Using the Stanford SST Sentiment Dataset
**File**: [Stanford SST Sentiment Dataset.ipynb](./Assignment%20%233/Text%20Classification%20Using%20the%20Stanford%20SST%20Sentiment%20Dataset.ipynb)  
Assignment 3 focuses on the Stanford SST Sentiment Dataset, which categorizes movie review sentences as positive, negative, or neutral. This sentiment analysis project has applications for businesses, investors, and content creators.

**Models Tested**:
1. Simple LSTM model
2. 1D CNN with GlobalMaxPooling
3. LSTM with pre-trained GloVe embeddings
4. Bidirectional LSTM
5. CNN with multiple filter sizes
6. Bidirectional LSTM with GloVe embeddings, dropout, and recurrent dropout

Models 3 and 6 achieved the highest accuracy (0.8057 and 0.8299). The use of **pre-trained GloVe embeddings** and **Bidirectional layers** in Model 6 significantly improved performance, showcasing the value of advanced techniques in sentiment analysis.
