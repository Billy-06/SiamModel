import tkinter as tk
from tkinter import filedialog
import os
import shutil

# Create the main window
root = tk.Tk()
root.title("Image Uploader")
root.geometry("350x200")

# Create a directory for storing images
folder_name = 'registered_faces'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Label for user ID entry
id_label = tk.Label(root, text="Enter Your ID:")
id_label.pack()

# Entry widget for user ID
user_id_entry = tk.Entry(root)
user_id_entry.pack()

# Function to upload image
def upload_image():
    user_id = user_id_entry.get().strip()
    if user_id:
        user_folder = os.path.join(folder_name, user_id)
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
        
        file_path = filedialog.askopenfilename()
        if file_path:
            shutil.copy(file_path, user_folder)
            tk.Label(root, text="Image Uploaded Successfully!").pack()
    else:
        tk.Label(root, text="Please enter a valid ID.").pack()

# Button to upload image
upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
upload_btn.pack(pady=20)

# Run the application
root.mainloop()
