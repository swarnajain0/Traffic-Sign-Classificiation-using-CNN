# Traffic-Sign-Classificiation-using-CNN
 Traffic Sign Recognition | CNN + TensorFlow + OpenCV | Trained on the GTSRB dataset with data augmentation, achieving 90%+ accuracy and real-time prediction from webcam feed.

📌 Features
- Preprocessing: Grayscale conversion, Histogram Equalization, Normalization  
- Data Augmentation for robust training  
- CNN with multiple convolution + pooling layers  
- Real-time classification using webcam (OpenCV)  
- Single image testing mode  
- Achieves ~90%+ accuracy after training

 📂 Project Structure
├── myData/ # dataset (not uploaded due to size)
├── labels.csv # class ID to name mapping
├── train.py # training script
├── test.py # real-time webcam test
├── model_trained.p # trained model (pickle format)
├── requirements.txt # required libraries
└── README.md # project documentation
