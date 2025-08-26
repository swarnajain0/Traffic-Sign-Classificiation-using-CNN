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


📊 Dataset
Dataset used: [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_news.html)  
Contains **43 classes** of traffic signs.  
⚠️ Not uploaded here due to size. Please download it separately.


## ⚙️ Installation
Clone this repo and install dependencies:
git clone https://github.com/<your-username>/Traffic-Sign-Classification.git
cd Traffic-Sign-Classification
pip install -r requirements.txt


for training: python train.py
for Real-Time Testing (Webcam): python test.py

📈 Results
Accuracy: ~90% after 15 epochs
Example Prediction: Stop (98.7%)

