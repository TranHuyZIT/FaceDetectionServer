import json
import cv2
import face_recognition

f = open('data.json','r+')

def load():
    data = json.load(f)
    return data

def write(data):
    string = json.dumps(data)
    f.write(string)

def process(image, name):
    data = load()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #BGR được chuyển đổi sang RGB
    encode = face_recognition.face_encodings(image)[0]

    print("Encode successfully! ")
    data.append({ "class_name": name, "encode": encode.tolist()})
    print(data)
    write(data)
