import json
import cv2
import face_recognition



def load():
    f = open('data.json', 'r+')
    data = json.load(f)
    f.close()
    return data


def write(data):
    f = open('data.json', 'w')
    string = json.dumps(data)
    f.write(string)
    f.close()


def process(image, name):
    data = load()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #BGR được chuyển đổi sang RGB
    encode = face_recognition.face_encodings(image)[0]
    print("Encode successfully! ")
    data.append({ "class_name": name, "encode": encode.tolist()})
    print(data)
    write(data)


def main():
  image = cv2.imread("hduy.png")
  process(image, "Hoang Duy")

main()
