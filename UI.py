import tkinter as tk
import cv2
import os
import pathlib
import Utilities_CV as u_cv

# initialise dataset path
data_dir = "./Shape_data/"
data_dir = pathlib.Path(data_dir)


class CVApp(tk.Tk):
    def __init__(self, *args, ** kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        # for F in (MainWindow, None):
        frame = MainWindow(container, self)
        self.frames[MainWindow] = frame
        frame.grid(row=0, column=0, sticky="nsew")

        self.showframe(MainWindow)

    def showframe(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class MainWindow(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self._controller = controller
        # instantiate vision class
        self.vision = u_cv.ComputerVision()

        self.directory = tk.StringVar()

        # additional ui details go here
        self.titlelbl = tk.Label(text="Geometric Shape Recognition")
        self.titlelbl.place(x=160, y=10)
        self.results_box = tk.Listbox(width=55, height=9)
        self.results_box.place(x=82, y=45)
        self.inputlbl = tk.Label(text="Upload image or get webcam feed")
        self.inputlbl.place(x=150, y=200)
        self.uploadlbl = tk.Label(text="Enter image url")
        self.uploadlbl.place(x=95, y=230)
        self.uploadent = tk.Entry(width=30, bg="white", textvariable=self.directory)
        self.uploadent.place(x=50, y=255)
        self.uploadbtn = tk.Button(text="Get Image Folder", width=20, height=1, bg="white", command=self.getDirectory)
        self.uploadbtn.place(x=80, y=285)
        self.cambtn = tk.Button(text="Get Webcam", width=10, height=1, bg="white", command=self.getWebcam)
        self.cambtn.place(x=350, y=250)

    def getWebcam(self):
        self.results_box.delete(0, 'end')
        print("Enter [q] to quit or [p] to make a prediction")
        screencap_count = 0
        index = 0
        self.results_box.insert(index, 'Results for Webcam Feed:')
        while True:
            ret = self.vision.get_cam()
            if not ret[0] and ret[1]:
                screencap_count += 1
                index += 1

                predictions = sorted(ret[1], key=lambda x: x[1], reverse=True)
                print(predictions)

                self.results_box.insert(index, 'Screenshot ' + str(screencap_count) + ' Results:')
                for item in predictions:
                    index += 1
                    self.results_box.insert(index, item)

            elif ret[0]:
                break

    def getDirectory(self):
        self.results_box.delete(0, 'end')
        index = 0
        self.results_box.insert(index, 'Results for Image Files:')

        direct = self.uploadent.get()
        # files = glob.glob(direct, recursive=True)
        dir_list = []
        file_list = []
        for root, dirs, files in os.walk(direct):
            for file in files:
                dir_list.append(os.path.join(root, file))
                file_list.append(file)
                print("File: " + file)

        if dir_list:
            print("Reading images from '", direct, "'...")
            for file in range(len(dir_list)):
                print("Prediction for ", dir_list[file], ": ")
                img = cv2.imread(dir_list[file])

                predictions = self.vision.get_prediction(img)
                predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
                print(predictions)

                index += 1

                self.results_box.insert(index, 'Image File: ' + file_list[file])
                for item in predictions:
                    index += 1
                    self.results_box.insert(index, item)

                # resize image for display
                img = cv2.resize(img, (350, 350))
                cv2.imshow('image', img)
                cv2.waitKey(10000)


def close():
    CVApp.quit(app)


app = CVApp()

app.geometry("500x350")
app.resizable(width=False, height=False)
app.title("Main")

app.mainloop()
