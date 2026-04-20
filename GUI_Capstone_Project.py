import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np

# Load model
model = pickle.load(open("churn_model.pkl", "rb"))

def predict():
    try:
        tenure = float(entry1.get())
        monthly = float(entry2.get())

        input_data = np.array([[tenure, monthly]])
        result = model.predict(input_data)

        if result[0] == 1:
            label_result.config(text="⚠ Customer Will Leave", fg="red")
        else:
            label_result.config(text="✅ Customer Will Stay", fg="green")

    except:
        messagebox.showerror("Error", "Please enter valid numbers")

# Window
root = tk.Tk()
root.title("Customer Churn Prediction System")
root.geometry("400x350")
root.configure(bg="#f0f4f7")

# Title
title = tk.Label(root, text="Customer Churn Predictor",
                 font=("Arial", 16, "bold"), bg="#f0f4f7", fg="#333")
title.pack(pady=15)

# Tenure
tk.Label(root, text="Tenure (Months)", font=("Arial", 11),
         bg="#f0f4f7").pack(pady=5)
entry1 = tk.Entry(root, font=("Arial", 11), justify="center")
entry1.pack(pady=5)

# Monthly Charges
tk.Label(root, text="Monthly Charges (₹)", font=("Arial", 11),
         bg="#f0f4f7").pack(pady=5)
entry2 = tk.Entry(root, font=("Arial", 11), justify="center")
entry2.pack(pady=5)

# Button
tk.Button(root, text="Predict",
          font=("Arial", 11, "bold"),
          bg="#4CAF50", fg="white",
          padx=10, pady=5,
          command=predict).pack(pady=15)

# Result
label_result = tk.Label(root, text="", font=("Arial", 12, "bold"),
                        bg="#f0f4f7")
label_result.pack(pady=10)

root.mainloop()
