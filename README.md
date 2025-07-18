# 🧠 Parkinson’s Wearing-Off Detection — Forecasting, App Design, and Behavior Feedback

**Author:** Kanika Gupta  
**Institution:** Kyushu Institute of Technology  
**Internship Project | 2025**

---

## 📍 Project Overview

This project addresses the **Wearing-Off (WO) phenomenon** in **Parkinson’s Disease (PD)** using data from **Garmin fitness trackers** and **smartphone apps**. It aims to forecast WO symptoms and present them through an **interactive mobile app** for clinicians and patients.

---

## 🎯 Objectives

- Improve baseline and advanced **prediction models**
- Compare **machine learning (ML)** and **deep learning (DL)** methods such as:
  - `LSTM`, `1D CNN`, `MLP`
  - `Random Forest`, `XGBoost`, `LightGBM`
- Predict WO **one hour ahead** using time-series behavioral data
- Display model outputs in a **mobile app** interface
- Collect **real-time user feedback** through FonLog

---

## 📊 Data Sources

- **Garmin Vivosmart 4**:  
  Tracks *heart rate (HR)*, *stress*, *steps*, *sleep patterns*
- **FonLog App**:  
  Gathers self-reported *WO episodes*, *drug intake times*, and *daily behavior logs*

---

## ⚙️ Technical Approach

- Preprocessing and **feature engineering** from wearable data
- **Label shifting**: Train models to predict WO at *t+1*
- **RandomOverSampler**: Handle data imbalance in binary WO classification
- Evaluation metrics:
  - **Balanced Accuracy**
  - **F1 Score**
  - **AUC (Area Under Curve)**

> 🚀 **Best-performing model**: `LSTM` with ~92% balanced accuracy

---

## 📱 App Integration

- **WoForecastProto**: A mobile prototype to visualize upcoming WO events
- **FonLog**: Allows patients to validate predictions with real feedback
- Backend includes a **trained server model** (ready for deployment)

---

## 🔮 Future Plans

- Conduct **real-time clinical trials**
- Enhance models using:
  - **Mood**, **Sleep**, **Stress**, and **Physical Activity**
- Extend the system to more patients and longer time spans

---

## 📚 References

- Victorino et al., 2021 & 2022 – Behavioral modeling of WO  
- Antonini et al., 2011 – WO scales and assessment  
- Colombo et al., 2015 – Patient-reported WO behavior  
- Lee et al., 2018 – Smartwatch-based Parkinson’s monitoring

---

*This project was completed as part of a research internship at Kyushu Institute of Technology. Special thanks to the Wearable Systems Lab and Dr. Shibata’s team.*
