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

## LOAD STATUS PREDICTION

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
    



    
