import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from keras.models import load_model
from classification import classify

from Siamese.SiameseModel import check_image
# Ensure the correct path is added for your predict_class function
sys.path.append('C:/Users/Mumba Ntambo/Documents/GitHub/SiamModel')
from antispoof_m import predict_class

# Additional function to ensure directory permissions
def ensure_directory_permissions(directory_path):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
        except PermissionError as e:
            messagebox.showerror("Permission Error", f"Permission denied while creating the directory: {e}")
            return False
    try:
        # Check if the directory is writable, and if not, change the permissions
        if not os.access(directory_path, os.W_OK):
            os.chmod(directory_path, stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    except Exception as e:
        messagebox.showerror("Permission Error", f"An error occurred while checking/modifying permissions: {e}")
        return False
    return True


class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.anti_spoofing_model = self.load_anti_spoofing_model('C:/Users/Mumba Ntambo/Documents/GitHub/SiamModel/antispoof.h5')
        self.root.geometry('900x700')

        # Directory for storing registered faces
        self.registered_faces_dir = r'registered_faces'
        if not os.path.exists(self.registered_faces_dir):
            os.makedirs(self.registered_faces_dir)
            
            

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', background='light gray', font=('Arial', 12))
        style.configure('TLabelFrame', background='light gray', font=('Arial', 12, 'bold'))
        style.configure('TButton', font=('Arial', 12, 'bold'))
        style.configure('TEntry', font=('Arial', 12))

        self.root.configure(bg='light gray')

        self.current_face_image = None
        self.video_stream_active = False

        self.setup_ui()
        self.start_video_stream()

    def load_anti_spoofing_model(self, model_path):
        try:
            return load_model(model_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load anti-spoofing model: {e}")
            return None

    def setup_ui(self):
        registration_frame = ttk.LabelFrame(self.root, text="Registration", padding=10)
        registration_frame.pack(padx=10, pady=5, fill="both", expand="yes")

        upload_btn = ttk.Button(registration_frame, text="Upload Image for Registration", command=self.upload_image_for_registration)
        upload_btn.pack(pady=5, fill="x")

        student_id_label = ttk.Label(registration_frame, text="Enter Student ID:")
        student_id_label.pack(pady=5)

        self.student_id_entry = ttk.Entry(registration_frame)
        self.student_id_entry.pack(pady=5, fill="x")

        register_btn = ttk.Button(registration_frame, text="Register Face", command=self.register_face)
        register_btn.pack(pady=5, fill="x")

        sign_in_frame = ttk.LabelFrame(self.root, text="Sign In", padding=10)
        sign_in_frame.pack(padx=10, pady=5, fill="both", expand="yes")

        capture_btn = ttk.Button(sign_in_frame, text="Capture Image for Sign In", command=self.capture_image_for_sign_in)
        capture_btn.pack(pady=5, fill="x")

        self.image_label = ttk.Label(self.root)
        self.image_label.pack(pady=5)

        self.result_label = ttk.Label(self.root, text="Result: Unknown", font=("Arial", 14, 'bold'))
        self.result_label.pack(pady=5)

    def start_video_stream(self):
        self.cap = cv2.VideoCapture(0)
        self.video_stream_active = True
        self.update_video_stream()

    def update_video_stream(self):
        if self.video_stream_active:
            ret, frame = self.cap.read()
            if ret:
                self.display_image(frame)
            self.root.after(10, self.update_video_stream)

    def upload_image_for_registration(self):
        file_types = [('JPEG Images', '*.jpeg'), ('PNG Images', '*.png'), ('JPG Images', '*.jpg')]
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=file_types)
        print(f"Selected file: {file_path}")  # Debugging line

        if not file_path:
            messagebox.showinfo("Info", "No file selected")
            return

        # Attempt to open the image using PIL first to check if it's valid
        try:
            from PIL import Image
            pil_image = Image.open(file_path)
            pil_image.verify()  # This will raise an exception if the file is not a valid image
            pil_image.close()  # Close the PIL image since we only wanted to verify it
        except (IOError, SyntaxError) as e:
            messagebox.showerror("Error", f"The file is not a valid image or is corrupted: {e}")
            return

    # If PIL verification passed, we proceed with OpenCV
        self.current_face_image = cv2.imread(file_path)
        if self.current_face_image is None:
            messagebox.showerror("Error", "OpenCV failed to load image. Please ensure the file is a valid image format and not corrupted.")
            return

        print(f"Image loaded successfully: {self.current_face_image is not None}")  # Debugging line
        self.display_image(self.current_face_image)






    def display_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        self.image_label.config(image=image)
        self.image_label.image = image



    def register_face(self):
        student_id = self.student_id_entry.get()
        if not student_id:
            messagebox.showerror("Error", "Please enter the student ID")
            return

        if self.current_face_image is None:
            messagebox.showerror("Error", "No image uploaded for registration")
            return

        # Using raw string for the folder path
        student_folder = os.path.join(self.registered_faces_dir, student_id)

        # Check and create the directory if it doesn't exist
        if not os.path.exists(student_folder):
            os.makedirs(student_folder)

        preprocessed_image = self.preprocess_image_for_model(self.current_face_image)
        if preprocessed_image is None or preprocessed_image.size == 0:
            messagebox.showerror("Error", "Invalid image data")
            return

        temp_image_path = os.path.join(student_folder, f"{student_id}.jpg")

        try:
            cv2.imwrite(temp_image_path, preprocessed_image)
            messagebox.showinfo("Success", f"Face registered successfully for Student ID: {student_id}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {e}")


        # try:
        #     if not self.is_real_face(temp_image_path):
        #         messagebox.showerror("Error", "Spoofing detected! Registration denied.")
        #         os.remove(temp_image_path)
        #         return
        # except Exception as e:
        #     messagebox.showerror("Error", f"Error in anti-spoofing check: {e}")
        #     return

        messagebox.showinfo("Success", f"Face registered successfully for Student ID: {student_id}")



    def get_face_data(self, student_id):
        image_path = os.path.join(self.registered_faces_dir, f"{student_id}.jpg")
        if os.path.exists(image_path):
            return cv2.imread(image_path)
        else:
            return None

    def capture_image_for_sign_in(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_face_image = frame
                self.save_image_to_root_folder(frame)
                self.sign_in()

    def save_image_to_root_folder(self, image):
        self.root_directory = os.getcwd()
        self.captured_image_path = os.path.join(self.root_directory, 'captured_face.jpg')
        cv2.imwrite(self.captured_image_path, image)
        print(f"Image saved to {self.captured_image_path}")

    def sign_in(self):
        if self.current_face_image is None:
            messagebox.showerror("Error", "No image captured for sign-in")
            return

        if not self.is_real_face(self.captured_image_path):
          messagebox.showerror("Error", "Spoofing detected! Access denied.")
        #   return
        
        else:
            cap_img_path = os.path.join(self.root_directory, 'captured_face.jpg')
            # Path to the classified image
            label = classify(self.current_face_image)
            messagebox.showinfo("Info", f"This is: {label}")
            cls_img_path = os.path.join(self.root_directory, 'rg1.jpeg')
            
            messagebox.showinfo("Info", "Processing Image...")
            self.confirm_identity(cap_img_path,cls_img_path)
            
            

        student_id_for_sign_in = self.student_id_entry.get()  # Assume student ID is entered for sign-in
        registered_face_data = self.get_face_data(student_id_for_sign_in)

        if registered_face_data is not None:
            registered_image_path = os.path.join(self.registered_faces_dir, f"{student_id_for_sign_in}.jpg")
            if self.confirm_identity(self.captured_image_path, registered_image_path):
                messagebox.showinfo("Success", "Sign in successful.")
            else:
                messagebox.showerror("Error", "Face match failed. Sign in unsuccessful.")
        else:
            messagebox.showerror("Error", "Student ID not registered")
            

    def is_real_face(self, image_path):
        prediction = predict_class(image_path)
        return prediction == "real"

    def load_anti_spoofing_model(self, model_path):
        return load_model(model_path)

    def confirm_identity(self, image_path_one, image_path_two):
        value = check_image(image_path_one, image_path_two)
        if value < 0.5:
            messagebox.showerror("Error", "Not a Match")
            print("No-match")
        else:
            messagebox.showinfo("Info", "Image Matched")
            print("Match")

    def preprocess_image_for_model(self, image):
        resized_image = cv2.resize(image, (160, 160))
        resized_image = resized_image.astype("float32")
        resized_image = np.expand_dims(resized_image, axis=0)
        return resized_image

if __name__ == "__main__":
    root = tk.Tk()
    gui = FaceRecognitionGUI(root)
    root.mainloop()