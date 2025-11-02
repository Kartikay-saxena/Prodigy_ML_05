# Prodigy_ML_05
Develop a model that can accurately recognize food items from images and estimate their calorie content, enabling users to track their dietary intake and make informed food choices.
  
Food Recognition using SVM (Food-101 Mini Version)

This repository contains my implementation for Task 05 of the Prodigy InfoTech 
Machine Learning Internship. The task required working with the Food-101 dataset, 
which is a very large collection of food images. To make experimentation faster, 
I used a smaller number of samples from each class and resized them to 32×32 
grayscale images.

The main focus here was dataset handling and building a simple but efficient 
pipeline that could run quickly even on limited hardware.

---

What I Did
- Loaded Food-101 style dataset with subfolders  
- Ignored hidden files like `.DS_Store`  
- Resized images for faster processing  
- Flattened them into feature vectors  
- Trained a Linear SVM classifier  
- Evaluated accuracy and generated a full report  

---

## 📂 Project Files
- `task05_food_recognition.py` 
