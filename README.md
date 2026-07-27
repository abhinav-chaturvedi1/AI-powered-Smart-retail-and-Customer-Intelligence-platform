# AI-Powered Smart Retail & Customer Intelligence Platform

A Flask API integrating computer vision, NLP, and a chatbot for a retail/e-commerce use case: recognizing returning customers via face recognition, classifying product images, analyzing customer review sentiment, and answering FAQs via a chatbot.

**Student:** Abhinav Chaturvedi (23BCE11191, Application No. IN26012479)
**Institution:** VIT Bhopal — B.Tech CSE (Core)

## Features

| Module | Endpoint | Description |
|---|---|---|
| Face Recognition | `POST /recognize-face` | Recognizes a registered customer from an uploaded photo |
| Product Classifier | `POST /classify-product` | Classifies a clothing product image into 1 of 10 categories |
| Sentiment Analysis | `POST /analyze-sentiment` | Classifies review/feedback text as Positive/Neutral/Negative |
| Chatbot | `POST /chatbot` | Matches a customer message to an FAQ intent and replies |
| Dashboard | `GET /dashboard/stats` | Aggregate registered-customer and visit-log stats |

## Tech Stack

- **Backend:** Flask (Blueprints per module)
- **Computer Vision:** OpenCV (Haar Cascade face detection, LBPH face recognizer)
- **Deep Learning:** TensorFlow/Keras (custom CNN for product classification)
- **NLP:** NLTK (preprocessing), scikit-learn (TF-IDF + Logistic Regression for both sentiment and chatbot intent classification)
- **Testing:** pytest with Flask's test client

## Project Structure

```
smart-retail-ai/
├── app/
│   ├── main.py                  # Flask entrypoint, loads all models at startup
│   ├── routers/                 # vision.py, nlp.py, chatbot.py — Blueprints
│   ├── services/                # cv_service, face_recognition_module,
│   │                             #   nlp_service, chatbot_service, training scripts
│   └── models/                  # trained model artifacts (.h5, .pkl, .xml)
├── notebooks/                   # EDA + training notebooks (01-04)
├── data/                        # datasets + intents.json + sample face photos
├── tests/                       # pytest integration tests
├── requirements.txt
└── report.md                    # full write-up (source for submission.pdf)
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m pytest tests/ -v     # run the test suite
python app/main.py             # start the API on localhost:5000
```

## Key Results

- **Product classifier:** 90.4% test accuracy (custom CNN, 8 epochs, Fashion-MNIST)
- **Sentiment analysis:** 76.9% test accuracy overall (weighted for class imbalance; Positive F1 0.89, Neutral F1 0.36 — see report for full breakdown)
- **Chatbot:** 11/12 sample conversations correctly classified across 18 intents
- **Face recognition:** correctly identifies both registered customers
- **API integration:** 9/9 pytest integration tests passing

## Notable Design Decisions

A few points where the implementation deviated from the original spec, documented here for transparency (also covered in detail in `report.md`):

- **Face recognition** uses OpenCV's LBPH recognizer instead of the dlib-based `face_recognition` library — `dlib` requires compiling from source, which was impractically slow in the development environment. LBPH is an explicitly valid fallback for this exact scenario.
- **Product classifier** uses a compact custom CNN trained from scratch instead of MobileNetV2 transfer learning — the pretrained ImageNet weights are hosted on a URL that was unreachable in the development sandbox, and a native CNN is arguably a better fit for 28x28 grayscale images anyway (avoids artificial upsampling/channel-tiling).

## Ethics Note

The face recognition module processes biometric data. A production deployment would require explicit customer consent before capture, disclosed retention limits, a deletion mechanism, and testing for accuracy disparities across demographics and lighting conditions. All sample photos used in development were collected from consenting individuals for educational purposes only.
