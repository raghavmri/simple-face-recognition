import numpy as np
import cv2
import face_recognition
import pickle
import random
import string
import pyttsx3
import emoji
import speech_recognition as sr
from art import tprint


fileName = "db.dat"
hyfmt = "-"*40
botname = "Simple Bot"


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def speak(text: str):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def getUsers():
    with open(fileName, "r+b") as f:
        if len(f.read()) == 0:
            return []
        f.seek(0)
        users = pickle.load(f)
    return users


def displayUsers():
    print("-"*40)
    users = getUsers()
    n = 1
    print("ID \t Name")
    for user in users:
        print(f"{n} \t {user[0]}")
        n += 1
    print("-"*40)


def deleteUser():
    print(hyfmt)

    try:
        displayUsers()
        n = int(input("Enter the ID of the user you want to delete: "))
        users = getUsers()
        users.pop(n-1)
        with open(fileName, "wb") as f:
            pickle.dump(users, f)
        print(f"User with ID {n} has been deleted successfully ✔")
    except Exception as e:
        print("There was an error deleting the user: ", e)
    print(hyfmt)


def newUser():
    try:
        print(hyfmt)
        userID = randomString()
        name = input("Enter your name: ")
        while True:
            print(emoji.emojize(
                "*"*10+" Please look at the camera :video_camera: "+"*"*10))
            video_capture = cv2.VideoCapture(0)
            ret, frame = video_capture.read()
            cv2.imwrite(f"images/{userID}.jpg", frame)
            video_capture.release()
            cv2.destroyAllWindows()

            image = face_recognition.load_image_file(f"images/{userID}.jpg")
            face_encoding = face_recognition.face_encodings(image)
            if len(face_encoding) == 0:
                print("No face detected, please try again")
                continue
            users = getUsers()
            users.append([name, face_encoding[0]])
            with open(fileName, "wb") as f:
                pickle.dump(users, f)
            break
        print(emoji.emojize(
            name+" has been added successfully ✔"))
        print(hyfmt)

    except Exception as e:
        print("Error: ", e)


def findUserUsingEncodings(encodings):
    users = getUsers()
    for user in users:
        # print(user)
        if np.array_equal(user[1], encodings):
            return user
    return None


def getUserInfo():

    video_capture = cv2.VideoCapture(0)

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        known_face_encodings = getUserEncodings()
        name = "unknown"
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = findUserUsingEncodings(
                    known_face_encodings[best_match_index])[0]

            face_names.append(name)

    process_this_frame = not process_this_frame
    video_capture.release()

    return face_names


def getUserEncodings():
    users = getUsers()
    return [user[1] for user in users]


def testFace():
    print(hyfmt)
    if getUsers() == []:
        print("No users found. Please add a user first.")
        print(hyfmt)
        return
    print("Looking for faces...")
    face_names = getUserInfo()
    if len(face_names) == 0:
        print("No face detected")
    else:
        print("Found: ", face_names[0])
    print(hyfmt)


def startBot():
    video_capture = cv2.VideoCapture(0)
    speak(f"{botname} is now online now")
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    greetedUsers = []
    process_this_frame = True
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        # Only process every other frame of video to save time
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations)
            face_names = []
            known_face_encodings = getUserEncodings()
            name = "unknown"
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = findUserUsingEncodings(
                        known_face_encodings[best_match_index])[0]
                if name != "unknown":
                    speak(f"Hello {name}")
                name = "unknown"

        process_this_frame = not process_this_frame

        # Display the resulting image
        # cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


r = sr.Recognizer()
if __name__ == "__main__":

    print("\n")
    tprint(f"{botname} THE SIMPLE BOT")
    while True:
        print("1. Start Bot \n2. Add User \n3. Display Users \n4. Delete User \n5. Test Face \n6. Exit")
        choice = input("Enter Your Choice: ")
        if choice == "1":
            startBot()
        elif choice == "2":
            newUser()
        elif choice == "3":
            displayUsers()
        elif choice == "4":
            deleteUser()
        elif choice == "5":
            testFace()
        elif choice == "6":
            tprint("Bye Bye!")
            break
        else:
            print("Invalid Choice! \n")
