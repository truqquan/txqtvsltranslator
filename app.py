import streamlit as st
import tensorflow as tf
from PIL import Image
# from tensorflow import keras
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import subprocess
from collections import OrderedDict
import paramiko
import shutil
import threading
import time
import speech_recognition as sr
from moviepy.editor import VideoFileClip, concatenate_videoclips

def combine_videos_vn(video_paths, output_path):
    video_clips = []
    folder_path = r"C:\Users\ACER\Downloads\TIN_HOC_TRE_THPT_TXQT\Streamlit\nlp\vietnamese"
    d = -1
    
    # Load the video clips
    for path in video_paths:
        d = d + 1

        file_path = os.path.join(folder_path, video_paths[d])

        if os.path.exists(file_path):
            path2 = "nlp/vietnamese/" + path
            video_clips.append(VideoFileClip(path2))

    # Concatenate the videos
    final_video = concatenate_videoclips(video_clips)

    # Save the final video
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac") 

def combine_videos_us(video_paths, output_path):
    video_clips = []
    folder_path = r"C:\Users\ACER\Downloads\TIN_HOC_TRE_THPT_TXQT\Streamlit\nlp\english"
    d = -1
    
    # Load the video clips
    for path in video_paths:
        d = d + 1

        file_path = os.path.join(folder_path, video_paths[d])

        if os.path.exists(file_path):
            path2 = "nlp/english/" + path
            video_clips.append(VideoFileClip(path2))

    # Concatenate the videos
    final_video = concatenate_videoclips(video_clips)

    # Save the final video
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac") 

def tokennizer(text, dict):
    global video_paths
    print(text)

    input = text.split(" ")
    words = []
    video_paths = []
    s = 0
    st1 = ""

    while True:
        e = len(input)
        while e > s:
            tmp_word = input[s:e]
            is_word = ""
            for item in tmp_word:
                is_word += item + " "
            is_word = is_word[:-1]
            e -= 1

            if is_word.lower() in dict:
                words.append(is_word)
                break

            if e == s:
                words.append(is_word)
                break
        if e >= len(input):
            break

        if is_word:
            st1 = words[len(words) - 1].replace(" ", "_") + ".mp4"
            video_paths.append(st1)

        s = e + 1

def speech_to_text_vn():
    # Create a recognizer instance
    r = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        st.write("Đang ghi âm...")

        # Adjust the microphone for ambient noise
        r.adjust_for_ambient_noise(source)

        # Capture the audio input from the user
        audio = r.listen(source)

        st.write("Đang chuyển đổi giọng nói...")

        try:
            # Use the Google Web Speech API to recognize the audio
            text = r.recognize_google(audio, language='vi-VN')
            return text
        except sr.UnknownValueError:
            st.write("Không thể nhận dạng giọng nói!")
        except sr.RequestError as e:
            st.write(f"Error: {e}")

def speech_to_text_us():
    # Create a recognizer instance
    r = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        st.write("Recording...")

        # Adjust the microphone for ambient noise
        r.adjust_for_ambient_noise(source)

        # Capture the audio input from the user
        audio = r.listen(source)

        st.write("Translating speech...")

        try:
            # Use the Google Web Speech API to recognize the audio
            text = r.recognize_google(audio, language='en-US')
            return text
        except sr.UnknownValueError:
            st.write("Không thể nhận dạng giọng nói!")
        except sr.RequestError as e:
            st.write(f"Error: {e}")

def run_code_on_pi1():
    # Specify the SSH connection details for your Raspberry Pi
    hostname = "10.42.0.1"
    username = "vip"
    password = "24092007"

    # Specify the folder and the command you want to run on the Raspberry Pi
    folder = "/home/vip/Downloads/KHKT_2024"
    command = f"cd {folder} && python3 server1.py"

    # Create an SSH client and connect to the Raspberry Pi
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname, username=username, password=password)

    # Create an empty placeholder to display the output
    output_placeholder = st.empty()

    # Start an interactive shell session on the Raspberry Pi
    shell = ssh_client.invoke_shell()

    # Send the command to the Raspberry Pi
    shell.send(command + "\n")


    # Continuously read the output from the shell
    while not shell.exit_status_ready():

        if shell.recv_ready():
            output = shell.recv(1024).decode("utf-8")
            # Update the placeholder with the output
            output_placeholder.text(output)

            print("start")
            print(output)
            print("end")


            if ("break4" in output) or ("BREAKKK" in output) or ("breakkkkkkkk1" in output) or ("breakk2" in output) or ("break5" in output):
                print("breakkkk")
                break

    print("exit success")



    # Close the SSH connection
    ssh_client.close()

def run_code_on_pi1b():
    # Specify the SSH connection details for your Raspberry Pi
    hostname = "10.42.0.1"
    username = "vip"
    password = "24092007"

    # Specify the folder and the command you want to run on the Raspberry Pi
    folder = "/home/vip/Downloads/KHKT_2024"
    command = f"cd {folder} && python3 server1b.py"

    # Create an SSH client and connect to the Raspberry Pi
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname, username=username, password=password)

    # Create an empty placeholder to display the output
    output_placeholder = st.empty()

    # Start an interactive shell session on the Raspberry Pi
    shell = ssh_client.invoke_shell()

    # Send the command to the Raspberry Pi
    shell.send(command + "\n")


    # Continuously read the output from the shell
    while not shell.exit_status_ready():

        if shell.recv_ready():
            output = shell.recv(1024).decode("utf-8")
            # Update the placeholder with the output
            output_placeholder.text(output)

            print("start")
            print(output)
            print("end")


            if ("break4" in output) or ("BREAKKK" in output) or ("breakkkkkkkk1" in output) or ("breakk2" in output) or ("break5" in output):
                print("breakkkk")
                break

    print("exit success")



    # Close the SSH connection
    ssh_client.close()

def run_code_on_pi2():
    # Specify the SSH connection details for your Raspberry Pi
    hostname = "10.42.0.1"
    username = "vip"
    password = "24092007"

    # Specify the folder and the command you want to run on the Raspberry Pi
    folder = "/home/vip/Downloads/KHKT_2024"
    command = f"cd {folder} && python3 server2.py"

    # Create an SSH client and connect to the Raspberry Pi
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname, username=username, password=password)

    # Create an empty placeholder to display the output
    output_placeholder = st.empty()

    # Start an interactive shell session on the Raspberry Pi
    shell = ssh_client.invoke_shell()

    # Send the command to the Raspberry Pi
    shell.send(command + "\n")

    # Continuously read the output from the shell
    while not shell.exit_status_ready():

        if shell.recv_ready():
            output = shell.recv(1024).decode("utf-8")
            # Update the placeholder with the output
            output_placeholder.text(output)

            print("start")
            print(output)
            print("end")


            if ("break4" in output) or ("BREAKKK" in output) or ("breakkkkkkkk1" in output) or ("breakk2" in output) or ("break5" in output):
                print("breakkkk")
                break

    print("exit success")

    # Close the SSH connection
    ssh_client.close()


with st.sidebar: 
    image = Image.open("hand.png")
    st.image(image)
    st.title("Thiết bị chuyển ngữ hỗ trợ người câm điếc trong giao tiếp")
    choice = st.radio("Tùy chọn", ["Giới thiệu","Chuyển đổi cử chỉ","Chuyển đổi giọng nói","Thu thập cử chỉ","Huấn luyện mô hình","Backup"])

if choice == "Giới thiệu":
    st.info("Dự án tham gia Hội thi Tin học trẻ toàn quốc lần thứ XXX năm 2024.")
    st.info("Tác giả: Cao Trung Quân Lớp 11A1 Trường THPT Thị xã Quảng Trị.")
    image = Image.open("handsign.jpg")
    st.image(image)

if choice == "Chuyển đổi cử chỉ":
    st.title("Chuyển đổi cử chỉ")
    image = Image.open("handsign_convert.png")
    st.image(image)
    if st.button("Vietnamese"):
        run_code_on_pi1()
    elif st.button("English"):
        run_code_on_pi1b()

if choice == "Chuyển đổi giọng nói":
    st.title("Chuyển đổi giọng nói")
    image = Image.open("speech_convert.png")
    st.image(image)

    if st.button("Vietnamese"):
        dict_file = "nlp/vietnamese/vietnamese_dict.txt"
        dict = {}

        with open(dict_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    key, value = line.split(":")
                    dict[key.strip()] = int(value.strip())
        print(dict)

        video_paths = []
        output_path = "nlp/vietnamese/combined_video.mp4"

        text = speech_to_text_vn()
        string1 = text.lower()
        print(string1)
        tokennizer(string1, dict)

        print(video_paths)
        combine_videos_vn(video_paths, output_path)

        st.title("Giọng nói thành ký tự:")
        st.write(text)

        st.title("Giọng nói thành cử chỉ:")
        st.video("nlp/vietnamese/combined_video.mp4")

    elif st.button("English"):
        dict_file = "nlp/english/english_dict.txt"
        dict = {}

        with open(dict_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    key, value = line.split(":")
                    dict[key.strip()] = int(value.strip())
        print(dict)

        video_paths = []
        output_path = "nlp/english/combined_video.mp4"

        text = speech_to_text_us()
        string1 = text.lower()
        print(string1)
        tokennizer(string1, dict)

        print(video_paths)
        combine_videos_us(video_paths, output_path)

        st.title("Giọng nói thành ký tự:")
        st.write(text)

        st.title("Giọng nói thành cử chỉ:")
        st.video("nlp/english/combined_video.mp4")

if choice == "Thu thập cử chỉ":
    st.title("Thu thập cử chỉ")
    sign = st.text_input("Đặt tên cho cử chỉ:")
    command =  "python udp-data-collect.py -d " + sign + " -l " + sign 

    if st.button("Bật server thu thập cử chỉ"):
        run_code_on_pi2()
    if st.button("Lấy dữ liệu"):
        process = subprocess.Popen(command, shell=True)
    if st.button("Hoàn thành"):
        st.success("Xin vui lòng chờ... Dữ liệu đang được tăng cường!")

        folder_path_data = 'C:\\Users\\ACER\\Downloads\\TIN_HOC_TRE_THPT_TXQT\\Streamlit\\' + sign

        for index in range(10):
            # Define the augmentation parameters
            augmentation_factor = 2  # Number of augmented samples to generate
            noise_std = 0.01  # Standard deviation of the noise

            # Define the column indices to exclude from augmentation
            columns_to_exclude = [2,3,4,8,9,10,14,15,16,20,21,22,26,27,28,32,33,34,38,39,40,45,46,47,51,52,53,57,58,59,63,64,65,69,70,71] # Columns to exclude (zero-based index)
            # Iterate over the CSV files in the subfolder
            for root, dirs, files in os.walk(folder_path_data):
                for file in files:
                    if file.endswith('.csv'):
                        # Read the original CSV file
                        file_path = os.path.join(root, file)
                        original_data = pd.read_csv(file_path)

                        # Apply noise augmentation
                        augmented_data = original_data.copy()

             
                        #for _ in range(augmentation_factor):
                        noise = np.random.normal(-noise_std, noise_std)
                        #print(len(noise))
                        try:
                            for i in columns_to_exclude:
                                augmented_data.iloc[:, i] = augmented_data.iloc[:, i].astype(float) + noise


                            # Create a new file name for the augmented data
                            new_file_name = os.path.splitext(file)[0] + '1.csv'
                            new_file_path = os.path.join(root, new_file_name)

                            # Save the augmented data to a new CSV file
                            augmented_data.to_csv(new_file_path, index=False)

                            print(f"Augmented data saved to: {new_file_path}")
                        except:
                            pass

        source_folder = 'C:\\Users\\ACER\\Downloads\\TIN_HOC_TRE_THPT_TXQT\\Streamlit\\' + sign
        destination_folder =  'C:\\Users\\ACER\\Downloads\\TIN_HOC_TRE_THPT_TXQT\\Streamlit\\data'

        if os.path.isdir(source_folder) and os.path.isdir(destination_folder):
            try:
                shutil.move(source_folder, destination_folder)
                # st.success("Folder moved successfully.")
            except Exception as e:
                st.error(f"An error occurred while moving the folder: {str(e)}")
        else:
            st.error("Invalid folder paths.")

        st.success("Đã lấy dữ liệu thành công!")
        
if choice == "Huấn luyện mô hình":
    st.title("Huấn luyện mô hình")
    folder_path = st.text_input("Hãy nhập đường dẫn đến thư mục")
    if folder_path:
        # st.success("Chọn thư mục thành công!")
        st.write("Đang tải lên dữ liệu của bạn...")

        if 'Cập nhật' not in st.session_state:
            st.session_state['Cập nhật'] = False

        x = []
        y = []
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                if file_name.endswith(".csv"):
                    file_path = os.path.join(root, file_name)
                    label = file_name.split(".")[0]  # Extract the label from the file name

                    df = pd.read_csv(file_path)

                    # Process the data and extract features
                    # Assuming your first column is the timestamp and the remaining columns are the features
                    timestamps = df.iloc[:, 0]
                    features = df.iloc[:, 1:]  # Adjust the column index as per your data structure
                    feature_array = features.values.flatten()

                    if feature_array.dtype == np.float64 and feature_array.shape == (7400,):

                        # print(features.values.)

                        # Add the features and label to the lists
                        x.append(features.values.flatten())
                        y.append(label)
                    else:
                        #print("errorrrrrrrrrrrrrrrrrrrrrrrrrrr")
                        print("Error: Invalid features")
                        print(file_path)
                        #os.remove(file_path)

        label_list = list(OrderedDict.fromkeys(y))
        num_classes = len(label_list)

        # Convert the lists into numpy arrays
        x = np.array(x)
        y = np.array(y)

        # Perform label encoding
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Split the data into training and testing sets
        test_size = 0.2  # Percentage of data to use for testing (adjust as needed)
        x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=test_size, random_state=42)

        x_train = keras.utils.normalize(x_train, axis=1)
        x_test = keras.utils.normalize(x_test, axis=1)

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        # print(x_train[0].shape)
        input_shape = x_train[0].shape

        st.success("Đã tải dữ liệu thành công! Đang chuyển sang bước Huấn luyện mô hình.")
        ### TRAIN MODEL


        st.write("Đang huấn luyện mô hình máy học...")

        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(25, activation='relu'),
            
            keras.layers.Dense(num_classes, activation='softmax')  # Change activation to softmax
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


        print("Training model ...")

        # Train the model
        epochs = 30
        losses = []
        accuracies = []
        for epoch in range(epochs):
            # Perform model training for each epoch
            history = model.fit(x_train, y_train)
            
            # Display the epoch information in Streamlit
            losses.append(history.history['loss'][0])
            accuracies.append(history.history['accuracy'][0])

            st.write(f"Epoch {epoch+1}/{epochs} hoàn thành:")
            st.write(f"Sai số: {losses[-1]}, Độ chính xác: {accuracies[-1]}")
            st.write()

        # Evaluate model on test set
        score = model.evaluate(x_test, y_test, verbose=0)

        st.success("Mô hình đã được huấn luyện thành công!")
        st.write(f"Sai số: {score[0]}")
        st.write(f"Độ chính xác: {score[1]}")

        # Save the Keras model (If it's ok)
        model.save('my_model.h5')

        # Convert the Keras model to TensorFlow Lite format
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save the TensorFlow Lite model
        with open('my_model.tflite', 'wb') as f:
            f.write(tflite_model)

        # Convert model to float16
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        tflite_fp16_model = converter.convert()
        # Save the TensorFlow Lite float16 model
        with open('my_model.tflite', 'wb') as f:
            f.write(tflite_model)

        # Save the label_list as a text file if it doesn't exist
        # if not os.path.exists('label_list.txt'):
        with open('label_list.txt', 'w') as f:
            for label in label_list:
                f.write(label + '\n')

        # Define the file paths on your computer and Raspberry Pi
        local_model_file = r'C:\Users\ACER\Downloads\TIN_HOC_TRE_THPT_TXQT\Streamlit\my_model.tflite'
        local_label_file = r'C:\Users\ACER\Downloads\TIN_HOC_TRE_THPT_TXQT\Streamlit\label_list.txt'
        remote_model_file = '/home/vip/Downloads/KHKT_2024/my_model.tflite'
        remote_label_file = '/home/vip/Downloads/KHKT_2024/label_list.txt'

        # Define the Raspberry Pi's IP address, username, and password
        raspberry_pi_ip = '10.42.0.1'
        username = 'vip'
        password = '24092007'

        # Create an SSH client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            # Connect to the Raspberry Pi
            ssh_client.connect(raspberry_pi_ip, username=username, password=password)

            # Create an SFTP client over the SSH connection
            sftp_client = ssh_client.open_sftp()

            # Transfer the files
            sftp_client.put(local_model_file, remote_model_file)
            sftp_client.put(local_label_file, remote_label_file)

            print("Model on server are updated!")

            # Close the SFTP client
            sftp_client.close()

        finally:
            # Close the SSH client
            ssh_client.close()

        st.success("Đã cập nhật thành công vào thiết bị!")

if choice == "Backup":
    if st.button("Restart thiết bị!"):
        # Define the file paths on your computer and Raspberry Pi

        local_model_file = r'C:\Users\ACER\Downloads\TIN_HOC_TRE_THPT_TXQT\Streamlit\Backup\my_model.tflite'
        local_label_file = r'C:\Users\ACER\Downloads\TIN_HOC_TRE_THPT_TXQT\Streamlit\Backup\label_list.txt'
        remote_model_file = '/home/vip/Downloads/KHKT_2024/my_model.tflite'
        remote_label_file = '/home/vip/Downloads/KHKT_2024/label_list.txt'


        # Define the Raspberry Pi's IP address, username, and password
        raspberry_pi_ip = '10.42.0.1'
        username = 'vip'
        password = '24092007'

        # Create an SSH client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            # Connect to the Raspberry Pi
            ssh_client.connect(raspberry_pi_ip, username=username, password=password)

            # Create an SFTP client over the SSH connection
            sftp_client = ssh_client.open_sftp()

            # Transfer the files
            sftp_client.put(local_model_file, remote_model_file)
            sftp_client.put(local_label_file, remote_label_file)

            print("Model on server are updated!")

            # Close the SFTP client
            sftp_client.close()

        finally:
            # Close the SSH client
            ssh_client.close()

        st.success("Hoàn tất!")
