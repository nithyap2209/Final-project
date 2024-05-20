# Final-project
  * Eye Disease Classification, 
  * Loan Status Prediction(Classification model),
  * Sales Forecasting(Regression model), 
  * Credit Card Data: Marketing Strategy(Clustering)

# EYE DISEASE CLASSIFICATION
Eye disease classification is a research area that focuses on developing algorithms and models to accurately classify different types of eye diseases based on medical imaging data. It plays a critical role in assisting ophthalmologists and healthcare professionals in effectively diagnosing and treating eye diseases.

The primary objective of eye disease classification is to leverage machine learning and computer vision techniques to analyze medical images and detect the four diseases: cataract, diabetic retinopathy, glaucoma, normal.

- The dataset consists of Normal, Diabetic Retinopathy, Cataract and Glaucoma retinal images where each class have approximately 1000 images. These images are collected from various sorces like IDRiD, Oculur recognition, HRF etc.

- Diabetic retinopathy: Diabetic retinopathy is a complication of diabetes that affects the blood vessels in the retina. It can cause vision loss, including blurred or distorted vision, and in severe cases, lead to blindness. Early detection, regular eye exams, and proper management of diabetes are crucial for preventing and managing this condition.

- Cataracts: Cataract is a common age-related eye condition characterized by the clouding of the lens, leading to blurry vision and visual impairment. It can be treated surgically by replacing the cloudy lens with an artificial one, restoring clear vision and improving quality of life.

- Glaucoma: Glaucoma is a group of eye diseases that damage the optic nerve, often due to increased fluid pressure in the eye. It gradually leads to vision loss, starting with peripheral vision and potentially progressing to complete blindness. Timely diagnosis, treatment, and ongoing monitoring are vital for preserving vision and preventing irreversible damage.
  
## Import Libraries
    ### import system libs
    import os
    import time
    ### import data handling tools
    import cv2
    import numpy as np
    import pandas as pd
    from PIL import Image
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
    ### import Deep learning Libraries
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.metrics import categorical_crossentropy
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.layers import Flatten, Dense, Activation, GlobalAveragePooling2D
    ### Ignore Warnings
    import warnings
    warnings.filterwarnings("ignore")

### Class for Loading and Splitting Datasets
    class EyeDiseaseDataset:
        def __init__(self, dataDir):
            self.data_dir = dataDir
            
        def dataPaths(self):
            filepaths = []
            labels = []
            folds = os.listdir(self.data_dir)
            for fold in folds:
                foldPath = os.path.join(self.data_dir, fold)
                filelist = os.listdir(foldPath)
                for file in filelist:
                    fpath = os.path.join(foldPath, file)
                    filepaths.append(fpath)
                    labels.append(fold)
            return filepaths, labels
        
        def dataFrame(self, files, labels):
    
            Fseries = pd.Series(files, name='filepaths')
            Lseries = pd.Series(labels, name='labels')
            return pd.concat([Fseries, Lseries], axis=1)
       
        def split_(self):
            files, labels = self.dataPaths()
            df = self.dataFrame(files, labels)
            strat = df['labels']
            trainData, dummyData = train_test_split(df, train_size=0.8, shuffle=True, random_state=42, stratify=strat)
            strat = dummyData['labels']
            validData, testData = train_test_split(dummyData, train_size=0.5, shuffle=True, random_state=42, stratify=strat)
            return trainData, validData, testData


     dataDir=r"C:\Users\Selvam\OneDrive\Desktop\project\eye-diseases-classification\dataset"   
     
      
      dataSplit = EyeDiseaseDataset(dataDir)
    train_data, valid_data, test_data = dataSplit.split_()
    
      def display_random_image(df):
        random_row = df.sample(1).iloc[0]

    filepath = random_row['filepaths']
    label = random_row['labels']
    
    img = Image.open(filepath)
    plt.imshow(img)
    plt.title(f'Label:{label}')
    plt.axis('off')
    plt.show()

    display_random_image(train_data)
![image](https://github.com/nithyap2209/Final-project/assets/92367257/7c212ddd-e8a4-4ad4-8289-217eebe91a86)


### Function for Data Augmentation
    def augment_data( train_df, valid_df, test_df, batch_size=16):

      img_size = (256,256)
      channels = 3
      color = 'rgb'
      

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
              rotation_range=30,
              horizontal_flip=True,
              vertical_flip=True,
              brightness_range=[0.5, 1.5])
          
    valid_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
          
    train_generator = train_datagen.flow_from_dataframe(
              train_df,
              x_col='filepaths',
              y_col='labels',
              target_size=img_size,
              color_mode=color,
              batch_size=batch_size,
              shuffle=True,
              class_mode='categorical'
          )
   
    print("Shape of augmented training images:", train_generator.image_shape)
          
    valid_generator = valid_test_datagen.flow_from_dataframe(
              valid_df,
              x_col='filepaths',
              y_col='labels',
              target_size=img_size,
              color_mode=color,
              batch_size=batch_size,
              shuffle=True,
              class_mode='categorical'
          )
         
    print("Shape of validation images:", valid_generator.image_shape)
          
    test_generator = valid_test_datagen.flow_from_dataframe(
              test_df,
              x_col='filepaths',
              y_col='labels',
              target_size=img_size,
              color_mode=color,
              batch_size=batch_size,
              shuffle=False,
              class_mode='categorical'
          )
          
    print("Shape of test images:", test_generator.image_shape)
          
    return train_generator, valid_generator, test_generator
    train_augmented, valid_augmented, test_augmented = augment_data(train_data, valid_data, test_data)
![image](https://github.com/nithyap2209/Final-project/assets/92367257/6c610de2-7e99-407e-93c2-9c68a689631a)

    def show_images(gen):
        g_dict = gen.class_indices        # defines dictionary {'class': index}
        classes = list(g_dict.keys())     # defines list of dictionary's kays (classes), classes names : string
        images, labels = next(gen)        # get a batch size samples from the generator
        length = len(labels)       
        sample = min(length, 20)   
        plt.figure(figsize= (15, 17))
        for i in range(sample):
            plt.subplot(5, 5, i + 1)
            image = images[i] / 255      
            plt.imshow(image)
            index = np.argmax(labels[i])  
            class_name = classes[index]  
            plt.title(class_name, color= 'blue', fontsize= 7 )
            plt.axis('off')
        plt.show()
    show_images(train_augmented)
![output](https://github.com/nithyap2209/Final-project/assets/92367257/53e8fd73-a808-4aab-aa79-5794fd540e53)

### Download and compile the model
    from tensorflow.keras.applications import EfficientNetB3
    from tensorflow.keras import regularizers
    
    classes = len(list(train_augmented.class_indices.keys()))
    
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu' , kernel_regularizer = regularizers.l2(0.01))(x)
    
    predictions = Dense(classes, activation='softmax', kernel_regularizer = regularizers.l2(0.01))(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
### Fit the model

    history = model.fit(
        train_augmented,
        epochs=15, 
        validation_data=valid_augmented,
        )
        
   ![image](https://github.com/nithyap2209/Final-project/assets/92367257/5f74ecc5-4a78-4c6c-8421-6d3fba46618d)


### Plot the Accuracy and loss
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    print("Training Accuracy:", train_accuracy[-1])
    print("Validation Accuracy:", val_accuracy[-1])
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
![image](https://github.com/nithyap2209/Final-project/assets/92367257/1accd5a5-d13e-43d6-8008-0ec12a0690f4)
    
![image](https://github.com/nithyap2209/Final-project/assets/92367257/73800eed-c71f-4750-9df9-23bf7e6f1575)
![image](https://github.com/nithyap2209/Final-project/assets/92367257/00a33e45-719f-4e7e-98c1-51d4af488a73)
### Display the Actual and Predicted images
    def plot_actual_vs_predicted(model, test_data, num_samples=3):
        
        # Get a batch of test data
        test_images, test_labels = next(iter(test_data))
    
        predictions = model.predict(test_images)
    
        class_labels = list(train_augmented.class_indices.keys())
    
        sample_indices = np.random.choice(range(len(test_images)), num_samples, replace=False)
          # Plot the images with actual and predicted labels
        for i in sample_indices:
            actual_label = class_labels[np.argmax(test_labels[i])]
            predicted_label = class_labels[np.argmax(predictions[i])]
            plt.figure(figsize=(8, 4))
            # Actual Image
            plt.subplot(1, 2, 1)
            plt.imshow(test_images[i].astype(np.uint8))  
            plt.title(f'Actual: {actual_label}')
            plt.axis('off')
            # Predicted Image
            plt.subplot(1, 2, 2)
            plt.imshow(test_images[i].astype(np.uint8))  
            plt.title(f'Predicted: {predicted_label}')
            plt.axis('off')
            plt.show()
    plot_actual_vs_predicted(model, test_augmented)
![image](https://github.com/nithyap2209/Final-project/assets/92367257/3203101d-b681-4dd5-abb6-948a0a1e13e9)

![image](https://github.com/nithyap2209/Final-project/assets/92367257/8beaee3a-eaa9-43af-acc5-3613d379aa18)
![image](https://github.com/nithyap2209/Final-project/assets/92367257/efcc2afa-8a01-4f03-a351-c52a23db7298)
![image](https://github.com/nithyap2209/Final-project/assets/92367257/9205802c-8e56-49ee-bffc-e4002c94fef5)

### Confusion Matrix
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
    
    # Predict the classes for the test data
    test_labels = test_augmented.classes
    predictions = model.predict(test_augmented)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get the class labels
    class_labels = list(train_augmented.class_indices.keys())
    
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(test_labels, predicted_classes)
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Print classification report
    print(classification_report(test_labels, predicted_classes, target_names=class_labels))
![image](https://github.com/nithyap2209/Final-project/assets/92367257/73d693cf-962a-43b6-81a1-2b9ff34d1199)

![image](https://github.com/nithyap2209/Final-project/assets/92367257/89a3adf5-a909-444e-a3a9-90fbedc6e263)


![image](https://github.com/nithyap2209/Final-project/assets/92367257/38a6a01d-1484-4c63-8cb4-a724b6171f37)



# LOAN STATUS PREDICTION

### IMPORT LIBRARY 
    import pandas as pd  ##for data manipulation
    import numpy as np   # for linear algebra 
    import matplotlib.pyplot as plt #for creating static, interactive, and animated visualizations in various formats. 
    import seaborn as sns  # Python data visualization library
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    import matplotlib.image as img
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report # for Precision and Recall Analysis
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB

### LOAD AND READ DATA 
    data=pd.read_csv(r"C:\Users\Selvam\OneDrive\Desktop\project\Loan Status Prediction\loan_data.csv")
### DATA EXPLORATION
    data.shape
    data.head()
    data.columns
    data.info()
    data.isna().sum()
    data.dropna(inplace=True)
    data.isna().sum()
    data.tail() 
### DATA VISUALIZATION
    sns.pairplot(data)
   ![image](https://github.com/nithyap2209/Final-project/assets/92367257/12f7b980-e041-4717-a4ce-c3c30945fb90)

    sns.scatterplot(data = data , x='ApplicantIncome' , y='Loan_Amount_Term')
  ![image](https://github.com/nithyap2209/Final-project/assets/92367257/c73b6933-d4e0-4ce9-ac18-31b8a5aa9e17)
  
    sns.countplot(data=data,x='Gender')
    #  countplot for Frequency Counts & Data Exploration
  
  ![image](https://github.com/nithyap2209/Final-project/assets/92367257/e089825e-9d13-42eb-bc61-39cd915be51a)
  

    data['Married'].hist(bins=30) # histplot  for Visualizing Data Distribution 

![image](https://github.com/nithyap2209/Final-project/assets/92367257/9290d6b5-05b8-4f6b-86dc-b8b71b2e9f2e)

    sns.displot(data['Education'])
 
 ![image](https://github.com/nithyap2209/Final-project/assets/92367257/abdf156c-1f14-40e1-b236-7fdee43b1877)
   
    Income = data.groupby("Dependents")["ApplicantIncome"].mean()
    plt.plot(Income.index, Income.values, color="deepskyblue", linewidth=6)
    plt.title("The Dance of Dependents Over ApplicantIncome", fontsize=22, fontweight="bold")
    plt.show()

![image](https://github.com/nithyap2209/Final-project/assets/92367257/9b12a15b-f478-477b-b381-0b42f81d0121)

 
### DATA PREPROCESSING 
    x = data.drop('Loan_Status',axis=1)
    y = data['Loan_Status']
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    for col in x.columns:
        x[col] = label_encoder.fit_transform(x[col]) ## HANDEL X 
    x.head()
    y.unique()
    mapper_loan={'N':0,'Y':1}
    y=y.map(mapper_loan) # HANDEL Y 
    y.head()
### SPLIT DATA

     x_train , x_test , y_train , y_test = train_test_split (x,y,test_size = .3 ,shuffle = True , random_state =42)
     print("x_train shape = ", x_train.shape)
     print("y_train shape = ", y_train.shape)
     print("x_test shape = ", x_test.shape)
     print("y_test shape = ", y_test.shape)
     shapes = {
         'X_train': x_train.shape[0],
         'y_train': y_train.shape[0],
         'X_test': x_test.shape[0],
         'y_test': y_test.shape[0]
     }
     plt.figure(figsize=(10, 6))
     plt.bar(shapes.keys(), shapes.values())
     plt.xlabel('Datasets')
     plt.ylabel('Number of instances')
     plt.title('Distribution of Training and Validation Sets')
     plt.show()

![image](https://github.com/nithyap2209/Final-project/assets/92367257/83ed747e-150e-4cf2-979c-e808c4944042)
     
### DATA SCSLING 
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
### LOGISTIC REGRESSION MODEL
    lr_model=LogisticRegression() # call model
    
    lr_model.fit(x_train,y_train)
    y_pred=lr_model.predict(x_test)
    y_pred
    con= confusion_matrix(y_test,y_pred) # Evaluation of Model Performance & Sensitivity and Specificity Analysis
    sns.heatmap(con, annot=True, cmap='viridis', cbar=True) # heatmap for Matrix Data Representation

![image](https://github.com/nithyap2209/Final-project/assets/92367257/df174e41-06aa-4aed-9959-6605d3b0c869)
    
    sns.clustermap(con, annot=True, cmap='viridis', cbar=True) # clustermap for Discovering Patterns and Relationships

![image](https://github.com/nithyap2209/Final-project/assets/92367257/baa9d09e-44db-42e6-ace0-91d357d4351c)
    
    print("classification_report is ",classification_report(y_test ,y_pred)) 

![image](https://github.com/nithyap2209/Final-project/assets/92367257/6012ef7a-3daf-4512-8607-5288811d6992)
    
### KNN MODEL
    knn_model = KNeighborsClassifier()
    knn_model.fit(x_train, y_train)
    knn_pred = knn_model.predict(x_test)
    knn_pred
    print(classification_report(y_test, knn_pred))

![image](https://github.com/nithyap2209/Final-project/assets/92367257/d90434b6-8910-4012-bcc4-5f315ba0773f)
    
### GAUSSIANNB MODEL
    nb_model = GaussianNB()
    nb_model.fit(x_train, y_train)
    y_pred = nb_model.predict(x_test)
    y_pred
    print(classification_report(y_test, y_pred))

  ![image](https://github.com/nithyap2209/Final-project/assets/92367257/76e2113e-f1a7-46b7-a25e-39b826f87ee4)

## **Enhance Model Performance**
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.model_selection import GridSearchCV
    parameters = [{'penalty':['l1','l2']}, 
                  {'C':[1, 10, 100, 1000]}]
    grid_search = GridSearchCV(lr_model,
                               parameters,
                               cv = 5,
                               verbose=0)


    grid_search.fit(x_train, y_train)
    print("classification_report is ",classification_report(y_test ,y_pred)) 

![image](https://github.com/nithyap2209/Final-project/assets/92367257/ad36ecc3-a429-4ee3-8d87-51400d707c74)



#  SALES FORECASTING

   ## Import libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    # Ignore warnings
    import warnings
    warnings.filterwarnings("ignore")
    
## Load the datasets
    stores = pd.read_csv(r"C:\Users\Selvam\OneDrive\Desktop\project\Sales_Forecasting\Sales Forecasting\stores.csv")
    features = pd.read_csv(r"C:\Users\Selvam\OneDrive\Desktop\project\Sales_Forecasting\Sales Forecasting\features.csv")
    train = pd.read_csv(r"C:\Users\Selvam\OneDrive\Desktop\project\Sales_Forecasting\Sales Forecasting\train.csv")
## Display the first few rows of each dataset
    print(stores.head())
    print(features.head())
    print(train.head())
![image](https://github.com/nithyap2209/Final-project/assets/92367257/ab46f309-dd9e-4e1e-98fc-8a56c54c5e62)
    
    # Merge datasets on 'Store' and 'Date'
    data = train.merge(stores, on='Store')
    data = data.merge(features, on=['Store', 'Date', 'IsHoliday'])
    # Display the first few rows of the merged dataset
    print(data.head())
![image](https://github.com/nithyap2209/Final-project/assets/92367257/63bdf245-9aae-4d9e-a0e6-73541b38026c)

    # Convert 'Date' to datetime type
    data['Date'] = pd.to_datetime(data['Date'])

    # Extract year, month, and day from 'Date'
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day

    # Drop the 'Date' column as we have extracted useful features from it
    data = data.drop(columns=['Date'])

### Check for missing values
    print(data.isnull().sum())

    # Fill missing values for Markdown columns with 0 (assuming missing means no markdown)
    data.fillna(0, inplace=True)

    # Encode categorical features
    categorical_features = ['Store', 'Dept', 'Type', 'IsHoliday']
    numerical_features = data.drop(columns=categorical_features + ['Weekly_Sales']).columns

    # Preprocessing pipelines for numerical and categorical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

 ![image](https://github.com/nithyap2209/Final-project/assets/92367257/cdc04ee9-3bbe-44f1-9d13-7c280b8221f7)
       

### Define features and target variable
    X = data.drop(columns=['Weekly_Sales'])
    y = data['Weekly_Sales']

### Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

![image](https://github.com/nithyap2209/Final-project/assets/92367257/f0d65a1b-5b21-4362-8d60-0256100c90d6)
    
    # Use only 50% of the data
    X_train_small, _, y_train_small, _ = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
    
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))])
    
    model.fit(X_train_small, y_train_small)

## Make predictions
### Make predictions
    y_pred = model.predict(X_test)

### Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R^2 Score: {r2}')

### Plot actual vs predicted sales
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs Predicted Sales')
    plt.show()
    
![image](https://github.com/nithyap2209/Final-project/assets/92367257/7c116062-9bce-4849-9b63-2bf80287cbb0)

![image](https://github.com/nithyap2209/Final-project/assets/92367257/bd6af488-1a4e-45b3-8797-342840c033bc)
    


# CREDIT CARD CLUSTERING
## Project Description
The sample Dataset summarizes the usage behavior of about 9000 active credit card holders during the last 6 months. The file is at a customer level with 18 behavioral variables. You need to develop a customer segmentation to define marketing strategy from the dataset. 
## Objective
There are a lot of features in this dataset (18 behavioral features). We will now perform:<br>
* Data preprocessing<br>
* Clustering<br>
* Feature extraction to improve clustering<br>
* Experiment with various clustering models: KMeans, Agglomerative Hierarchical, Gaussian Mixture<br>
* Choosing the number of clusters<br>
* EDA to segment the customers<br>
* Concluding the project by giving marketing strategy based on what we learn from the data
## Data Description
Following is the Data Dictionary for Credit Card dataset:<br>
* CUST_ID: Identification of Credit Card holder (Categorical)<br>
* BALANCE: Balance amount left in their account to make purchases<br>
* BALANCE_FREQUENCY: How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)<br>
* PURCHASES: Amount of purchases made from account<br>
* ONEOFF_PURCHASES: Maximum purchase amount done in one-go<br>
* INSTALLMENTS_PURCHASES: Amount of purchase done in installment<br>
* CASH_ADVANCE: Cash in advance given by the user<br>
* PURCHASES_FREQUENCY: How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)<br>
* ONEOFFPURCHASESFREQUENCY: How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)<br>
* PURCHASESINSTALLMENTSFREQUENCY: How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)<br>
* CASHADVANCEFREQUENCY: How frequently the cash in advance being paid<br>
* CASHADVANCETRX: Number of Transactions made with "Cash in Advanced"<br>
* PURCHASES_TRX: Number of purchase transactions made<br>
* CREDIT_LIMIT: Limit of Credit Card for user<br>
* PAYMENTS: Amount of Payment done by user<br>
* MINIMUM_PAYMENTS: Minimum amount of payments made by user<br>
* PRCFULLPAYMENT: Percent of full payment paid by user<br>
* TENURE: Tenure of credit card service for user
### import necessary tools
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # preprocessing
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # ignore warnings
    import warnings
    warnings.filterwarnings(action="ignore")

    # clustering
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from matplotlib import cm
    from sklearn.metrics import silhouette_samples, silhouette_score
  ### load the data
    data = pd.read_csv(r"C:\Users\Selvam\OneDrive\Desktop\project\credit-card-clustering\Credit Card_Clustering.csv")
    # data overview
    print('Data shape: ' + str(data.shape))
    data.head()
    data.describe()
  ### Data Cleaning
  First, we check the missing/corrupted values.
    
    data.isna().sum()
  We will inpute these missing values with the median value.
  
         # inpute with median
         data.loc[(data['MINIMUM_PAYMENTS'].isnull()==True),'MINIMUM_PAYMENTS'] = data['MINIMUM_PAYMENTS'].median()
         data.loc[(data['CREDIT_LIMIT'].isnull()==True),'CREDIT_LIMIT'] = data['CREDIT_LIMIT'].median()
    # double check
    data.isna().sum()
  Now we drop CUST_ID column, then normalize the input values using StandardScaler().
    
    # drop ID column
    data = data.drop('CUST_ID', 1)

    # normalize values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled.shape
    
    data_imputed = pd.DataFrame(data_scaled, columns=data.columns)
   We should now be good to go for clustering.
## Clustering
### Correlation Check
    plt.figure(figsize = (12, 12))
    sns.heatmap(data_imputed.corr(), annot=True, cmap='coolwarm', 
                xticklabels=data_imputed.columns,
                yticklabels=data_imputed.columns)

![image](https://github.com/nithyap2209/Final-project/assets/92367257/f1352170-e383-41fe-9cea-888b18624483)

### Clustering using K-Means

In this section we will perform K-Means clustering on the data and check the clustering metrics (inertia, silhouette scores).

### Inertia Plot

First, we make the inertia plot:

      # inertia plotter function
      def inertia_plot(clust, X, start = 2, stop = 20):
          inertia = []
          for x in range(start,stop):
              km = clust(n_clusters = x)
              labels = km.fit_predict(X)
              inertia.append(km.inertia_)
          plt.figure(figsize = (12,6))
          plt.plot(range(start,stop), inertia, marker = 'o')
          plt.xlabel('Number of Clusters')
          plt.ylabel('Inertia')
          plt.title('Inertia plot with K')
          plt.xticks(list(range(start, stop)))
          plt.show()
          
      inertia_plot(KMeans, data_imputed)
      
![image](https://github.com/nithyap2209/Final-project/assets/92367257/242a075a-4a9e-411f-9fa4-4621d5069348)
      
Using the elbow method, we pick a good number of clusters to be 6.

### Silhouette Scores

Silhouette analysis can be used to study the separation distance between the resulting clusters. The silhouette plot displays a measure of how close each point in one cluster is to points in the neighboring clusters and thus provides a way to assess parameters like number of clusters visually. This measure has a range of [-1, 1].

We will now check the silhouette scores for different numbers of clusters. 

    def silh_samp_cluster(clust,  X, start=2, stop=5, metric = 'euclidean'):
        # taken from sebastian Raschkas book Python Machine Learning second edition
        for x in range(start, stop):
            km = clust(n_clusters = x)
            y_km = km.fit_predict(X)
            cluster_labels = np.unique(y_km)
            n_clusters = cluster_labels.shape[0]
            silhouette_vals = silhouette_samples(X, y_km, metric = metric)
            y_ax_lower, y_ax_upper =0,0
            yticks = []
            for i, c in enumerate(cluster_labels):
                c_silhouette_vals = silhouette_vals[y_km == c]
                c_silhouette_vals.sort()
                y_ax_upper += len(c_silhouette_vals)
                color = cm.jet(float(i)/n_clusters)
                plt.barh(range(y_ax_lower, y_ax_upper),
                        c_silhouette_vals,
                        height=1.0,
                        edgecolor='none',
                        color = color)
                yticks.append((y_ax_lower + y_ax_upper)/2.)
                y_ax_lower+= len(c_silhouette_vals)

        silhouette_avg = np.mean(silhouette_vals)
        plt.axvline(silhouette_avg,
                   color = 'red',
                   linestyle = "--")
        plt.yticks(yticks, cluster_labels+1)
        plt.ylabel("cluster")
        plt.xlabel('Silhouette Coefficient')
        plt.title('Silhouette for ' + str(x) + " Clusters")
        plt.show()

        
    for x in range(2, 7):
        alg = KMeans(n_clusters = x, )
        label = alg.fit_predict(data_imputed)
        print('Silhouette-Score for', x,  'Clusters: ', silhouette_score(data_imputed, label))
        
![image](https://github.com/nithyap2209/Final-project/assets/92367257/9a98156f-8c35-4f51-84f1-66e6b7b39c4f)

Silhouette plots:

    silh_samp_cluster(KMeans, data_imputed, stop=7)

![image](https://github.com/nithyap2209/Final-project/assets/92367257/ad76645e-48c7-47be-bef1-5cd42a41c339)

![image](https://github.com/nithyap2209/Final-project/assets/92367257/25b7449f-a3ac-4aac-ab4a-d16e8c286079)


![image](https://github.com/nithyap2209/Final-project/assets/92367257/fcb8acf0-b986-42d5-9cbf-d1a69028d557)


![image](https://github.com/nithyap2209/Final-project/assets/92367257/2f4ec94f-0ea4-482a-8b3a-83ed82d9ba14)


![image](https://github.com/nithyap2209/Final-project/assets/92367257/dc03dbf9-bd84-4b1d-8691-e05823395b41)


So far, we have a high average inertia, low silhouette scores, and very wide fluctuations in the size of the silhouette plots. This is not good. Let's apply feature extraction with PCA to improve clustering.

## Feature Extraction with PCA
### Clustering Metrics
Now we will apply PCA to improve clustering. We should be able to see lower inertias and higher silhouette scores after feature extraction.

    # apply PCA and display clustering metrics
    for y in range(2, 5):
        print("PCA with # of components: ", y)
        pca = PCA(n_components=y)
        data_p = pca.fit_transform(data_imputed)
        for x in range(2, 7):
            alg = KMeans(n_clusters = x, )
            label = alg.fit_predict(data_p)
            print('Silhouette-Score for', x,  'Clusters: ', silhouette_score(data_p, label) , '       Inertia: ',alg.inertia_)
        print()
        
![image](https://github.com/nithyap2209/Final-project/assets/92367257/91aeb396-ab37-4878-960c-dd8d2d26ac9c)

As you can see, 2 PCA components with 5-6 clusters would be our best bet. 

## Visualization
    data_p = pd.DataFrame(PCA(n_components = 2).fit_transform(data_imputed))
    preds = pd.Series(KMeans(n_clusters = 5,).fit_predict(data_p))
    data_p = pd.concat([data_p, preds], axis =1)
    data_p.columns = [0,1,'target']
    
    fig = plt.figure(figsize = (18, 7))
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']
    plt.subplot(121)
    plt.scatter(data_p[data_p['target']==0].iloc[:,0], data_p[data_p.target==0].iloc[:,1], c = colors[0], label = 'cluster 1')
    plt.scatter(data_p[data_p['target']==1].iloc[:,0], data_p[data_p.target==1].iloc[:,1], c = colors[1], label = 'cluster 2')
    plt.scatter(data_p[data_p['target']==2].iloc[:,0], data_p[data_p.target==2].iloc[:,1], c = colors[2], label = 'cluster 3')
    plt.scatter(data_p[data_p['target']==3].iloc[:,0], data_p[data_p.target==3].iloc[:,1], c = colors[3], label = 'cluster 4')
    plt.scatter(data_p[data_p['target']==4].iloc[:,0], data_p[data_p.target==4].iloc[:,1], c = colors[4], label = 'cluster 5')
    plt.legend()
    plt.title('KMeans Clustering with 5 Clusters')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    data_p = pd.DataFrame(PCA(n_components = 2).fit_transform(data_imputed))
    preds = pd.Series(KMeans(n_clusters = 6,).fit_predict(data_p))
    data_p = pd.concat([data_p, preds], axis =1)
    data_p.columns = [0,1,'target']
    
    plt.subplot(122)
    plt.scatter(data_p[data_p['target']==0].iloc[:,0], data_p[data_p.target==0].iloc[:,1], c = colors[0], label = 'cluster 1')
    plt.scatter(data_p[data_p['target']==1].iloc[:,0], data_p[data_p.target==1].iloc[:,1], c = colors[1], label = 'cluster 2')
    plt.scatter(data_p[data_p['target']==2].iloc[:,0], data_p[data_p.target==2].iloc[:,1], c = colors[2], label = 'cluster 3')
    plt.scatter(data_p[data_p['target']==3].iloc[:,0], data_p[data_p.target==3].iloc[:,1], c = colors[3], label = 'cluster 4')
    plt.scatter(data_p[data_p['target']==4].iloc[:,0], data_p[data_p.target==4].iloc[:,1], c = colors[4], label = 'cluster 5')
    plt.scatter(data_p[data_p['target']==5].iloc[:,0], data_p[data_p.target==5].iloc[:,1], c = colors[5], label = 'cluster 6')
    plt.legend()
    plt.title('KMeans Clustering with 6 Clusters')
    plt.xlabel('PC1')
    plt.ylabel('PC2')


![image](https://github.com/nithyap2209/Final-project/assets/92367257/2dd58d1c-c745-4c2b-a153-cfd1d0d1bf63)

So far, by applying PCA we have made notable improvement to KMeans model. Let's try other clustering models as well!

## Agglomerative Hierarchical Clustering with PCA
    data_p = pd.DataFrame(PCA(n_components = 2).fit_transform(data_imputed))
    preds = pd.Series(AgglomerativeClustering(n_clusters = 5,).fit_predict(data_p))
    data_p = pd.concat([data_p, preds], axis =1)
    data_p.columns = [0,1,'target']
    
    fig = plt.figure(figsize = (18, 7))
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']
    plt.subplot(121)
    plt.scatter(data_p[data_p['target']==0].iloc[:,0], data_p[data_p.target==0].iloc[:,1], c = colors[0], label = 'cluster 1')
    plt.scatter(data_p[data_p['target']==1].iloc[:,0], data_p[data_p.target==1].iloc[:,1], c = colors[1], label = 'cluster 2')
    plt.scatter(data_p[data_p['target']==2].iloc[:,0], data_p[data_p.target==2].iloc[:,1], c = colors[2], label = 'cluster 3')
    plt.scatter(data_p[data_p['target']==3].iloc[:,0], data_p[data_p.target==3].iloc[:,1], c = colors[3], label = 'cluster 4')
    plt.scatter(data_p[data_p['target']==4].iloc[:,0], data_p[data_p.target==4].iloc[:,1], c = colors[4], label = 'cluster 5')
    plt.legend()
    plt.title('Agglomerative Hierarchical Clustering with 5 Clusters')
    plt.xlabel('PC1')
    plt.ylabel('PC2')


    data_p = pd.DataFrame(PCA(n_components = 2).fit_transform(data_imputed))
    preds = pd.Series(AgglomerativeClustering(n_clusters = 6,).fit_predict(data_p))
    data_p = pd.concat([data_p, preds], axis =1)
    data_p.columns = [0,1,'target']
    
    plt.subplot(122)
    plt.scatter(data_p[data_p['target']==0].iloc[:,0], data_p[data_p.target==0].iloc[:,1], c = colors[0], label = 'cluster 1')
    plt.scatter(data_p[data_p['target']==1].iloc[:,0], data_p[data_p.target==1].iloc[:,1], c = colors[1], label = 'cluster 2')
    plt.scatter(data_p[data_p['target']==2].iloc[:,0], data_p[data_p.target==2].iloc[:,1], c = colors[2], label = 'cluster 3')
    plt.scatter(data_p[data_p['target']==3].iloc[:,0], data_p[data_p.target==3].iloc[:,1], c = colors[3], label = 'cluster 4')
    plt.scatter(data_p[data_p['target']==4].iloc[:,0], data_p[data_p.target==4].iloc[:,1], c = colors[4], label = 'cluster 5')
    plt.scatter(data_p[data_p['target']==5].iloc[:,0], data_p[data_p.target==5].iloc[:,1], c = colors[5], label = 'cluster 6')
    plt.legend()
    plt.title('Agglomerative Hierarchical Clustering with 6 Clusters')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
![image](https://github.com/nithyap2209/Final-project/assets/92367257/3ab1d2ef-3e6e-46f6-9428-ebc942fea485)
    
## Gaussian Mixture Clustering with PCA

    data_p = pd.DataFrame(PCA(n_components = 2).fit_transform(data_imputed))
    preds = pd.Series(GaussianMixture(n_components = 5,).fit_predict(data_p))
    data_p = pd.concat([data_p, preds], axis =1)
    data_p.columns = [0,1,'target']
    
    fig = plt.figure(figsize = (18, 7))
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']
    plt.subplot(121)
    plt.scatter(data_p[data_p['target']==0].iloc[:,0], data_p[data_p.target==0].iloc[:,1], c = colors[0], label = 'cluster 1')
    plt.scatter(data_p[data_p['target']==1].iloc[:,0], data_p[data_p.target==1].iloc[:,1], c = colors[1], label = 'cluster 2')
    plt.scatter(data_p[data_p['target']==2].iloc[:,0], data_p[data_p.target==2].iloc[:,1], c = colors[2], label = 'cluster 3')
    plt.scatter(data_p[data_p['target']==3].iloc[:,0], data_p[data_p.target==3].iloc[:,1], c = colors[3], label = 'cluster 4')
    plt.scatter(data_p[data_p['target']==4].iloc[:,0], data_p[data_p.target==4].iloc[:,1], c = colors[4], label = 'cluster 5')
    plt.legend()
    plt.title('Gaussian Mixture Clustering with 5 Clusters')
    plt.xlabel('PC1')
    plt.ylabel('PC2')


    data_p = pd.DataFrame(PCA(n_components = 2).fit_transform(data_imputed))
    preds = pd.Series(GaussianMixture(n_components = 6,).fit_predict(data_p))
    data_p = pd.concat([data_p, preds], axis =1)
    data_p.columns = [0,1,'target']
    
    plt.subplot(122)
    plt.scatter(data_p[data_p['target']==0].iloc[:,0], data_p[data_p.target==0].iloc[:,1], c = colors[0], label = 'cluster 1')
    plt.scatter(data_p[data_p['target']==1].iloc[:,0], data_p[data_p.target==1].iloc[:,1], c = colors[1], label = 'cluster 2')
    plt.scatter(data_p[data_p['target']==2].iloc[:,0], data_p[data_p.target==2].iloc[:,1], c = colors[2], label = 'cluster 3')
    plt.scatter(data_p[data_p['target']==3].iloc[:,0], data_p[data_p.target==3].iloc[:,1], c = colors[3], label = 'cluster 4')
    plt.scatter(data_p[data_p['target']==4].iloc[:,0], data_p[data_p.target==4].iloc[:,1], c = colors[4], label = 'cluster 5')
    plt.scatter(data_p[data_p['target']==5].iloc[:,0], data_p[data_p.target==5].iloc[:,1], c = colors[5], label = 'cluster 6')
    plt.legend()
    plt.title('Gaussian Mixture Clustering with 6 Clusters')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

![image](https://github.com/nithyap2209/Final-project/assets/92367257/16078df9-0748-4049-937e-22b94814d06b)
    
# Exploratory Data Analysis
We are picking 6 clusters for this EDA. Let's make a Seaborn pairplot with selected/best columns to show how the clusters are segmenting the samples:

    # select best columns
    best_cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS"]
    
    # dataframe with best columns
    data_final = pd.DataFrame(data_imputed[best_cols])
    
    print('New dataframe with best columns has just been created. Data shape: ' + str(data_final.shape))
    
    # apply KMeans clustering
    alg = KMeans(n_clusters = 6)
    label = alg.fit_predict(data_final)
    
    # create a 'cluster' column
    data_final['cluster'] = label
    best_cols.append('cluster')
    
    # make a Seaborn pairplot
    sns.pairplot(data_final[best_cols], hue='cluster')

  ![image](https://github.com/nithyap2209/Final-project/assets/92367257/968894e1-6e4e-4929-966a-9808de10bc91)

We can see some interesting correlations between features and clusters that we have made above. Let's get into detailed analysis.

## Cluster 0 (Blue): The Average Joe
    sns.pairplot(data_final[best_cols], hue='cluster', x_vars=['PURCHASES', 'PAYMENTS', 'CREDIT_LIMIT'],
                y_vars=['cluster'],
                height=5, aspect=1)

 ![image](https://github.com/nithyap2209/Final-project/assets/92367257/918ec995-33e3-4c80-8b93-9cd157f6c175)
               
This group of users, while having the highest number of users by far, is fairly frugal: they have lowest purchases, second lowest payments, and lowest credit limit. The bank would not make much profit from this group, so there should be some sorts of strategy to attract these people more.

## Cluster 1 (Orange): The Active Users
    sns.pairplot(data_final[best_cols], hue='cluster', x_vars=['PURCHASES', 'PAYMENTS', 'CREDIT_LIMIT'],
                y_vars=['cluster'],
                height=5, aspect=1)
 ![image](https://github.com/nithyap2209/Final-project/assets/92367257/fb1477c8-4529-453a-b869-dacee6ca2cb7)
               
This group of users is very active in general: they have second highest purchases, third highest payments, and the most varied credit limit values. This type of credit card users is the type you should spend the least time and effort on, as they are already the ideal one.

## Cluster 2 (Green): The Big Spenders

The Big Spenders. This group is by far the most interesting to analyze, since they do not only have the highest number of purchases, highest payments, highest minimum payments, but the other features are also wildly varied in values. Let's take a quick look at the pairplots.

        sns.pairplot(data_final[best_cols], hue='cluster', x_vars=['PURCHASES', 'PAYMENTS', 'CREDIT_LIMIT'], y_vars=['cluster'],
                    height=5, aspect=1)

 ![image](https://github.com/nithyap2209/Final-project/assets/92367257/c127dd69-bd22-4f8d-8685-d9e2c88d4368)
                   
As a nature of the "Big Spenders", there are many outliers in this cluster: people who have/make abnormally high balance, purchases, cash advance, and payment. The graph below will give you an impression of how outlier-heavy this cluster is - almost all the green dots are outliers relatively compared to the rest of the whole dataset.

    sns.pairplot(data_final[best_cols], hue='cluster', x_vars=['PURCHASES'], y_vars=['PAYMENTS'],
            height=5, aspect=1)

![image](https://github.com/nithyap2209/Final-project/assets/92367257/dacb2c1e-5726-4c22-b250-25a173cf5df1)
            
## Cluster 3 (Red): The Money Borrowers
    sns.pairplot(data_final[best_cols], hue='cluster', x_vars=['BALANCE', 'CASH_ADVANCE', 'PAYMENTS'],
            y_vars=['cluster'],
            height=5, aspect=1)

![image](https://github.com/nithyap2209/Final-project/assets/92367257/0dc9eb21-28e8-4120-a8a8-b0ffd6ef2e84)
            
Wildly varied balance, second highest payments, average purchases. The special thing about this cluster is that these people have the highest cash advance by far - there is even one extreme case that has like 25 cash advance points. We call these people "The Money Borrowers".
## Cluster 4 (Purple): The High Riskers
    sns.pairplot(data_final[best_cols], hue='cluster', x_vars=['MINIMUM_PAYMENTS'], y_vars=['CREDIT_LIMIT'],
            height=5, aspect=1)

 ![image](https://github.com/nithyap2209/Final-project/assets/92367257/37ccfaa0-f46d-4c0c-9d57-35bcd40b3090)
           
This group has absurdly high minimum payments while having the second lowest credit limit. It looks like the bank has identified them as higher risk.
## Cluster 5 (Brown): The Wildcards
    sns.pairplot(data_final[best_cols], hue='cluster', x_vars=['BALANCE'], y_vars=['CREDIT_LIMIT'],
            height=5, aspect=1)

![image](https://github.com/nithyap2209/Final-project/assets/92367257/532d797d-4d00-4338-874f-fe5790c92ef5)
            
This group is troublesome to analyze and to come up with a good marketing strategy towards, as both their credit limit and balance values are wildly varied. As you can see, the above graph looks like half of it was made of the color brown!

# Summary and Possible Marketing Strategy
We have learned a lot from this dataset by segmenting the customers into six smaller groups: the Average Joe, the Active Users, the Big Spenders, the Money Borrowers, the High Riskers, and the Wildcards. To conclude this cluster analysis, let's sum up what we have learned and some possible marketing strategies:<br>
* The Average Joe do not use credit card very much in their daily life. They have healthy finances and low debts. While encouraging these people to use credit cards more is necessary for the company's profit, business ethics and social responsibility should also be considered.<br>
<br>
* Identify active customers in order to apply proper marketing strategy towards them. These people are the main group that we should focus on.<br>
<br>
* Some people are just bad at finance management - for example, the Money Borrowers. This should not be taken lightly.<br>
<br>
* Although we are currently doing a good job at managing the High Riskers by giving them low credit limits, more marketing strategies targeting this group of customers should be considered.<br>

# Conclusion
In this project, we have performed data preprocessing, feature extraction with PCA, looked at various clustering metrics (inertias, silhouette scores), experimented with various Clustering algorithms (KMeans Clustering, Agglomerative Hierarchical Clustering, Gaussian Mixture Clustering), data visualizations, and business analytics. 
This project is also my first try on the business side of Data Science, and how we can use Machine Learning to solve practical, real life issues.




    
