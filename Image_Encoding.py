import face_recognition

image_of_person1 = face_recognition.load_image_file("Known_Persons/Nithin/image1.jpg")
# image_of_person2 = face_recognition.load_image_file("Known_Persons/nowfal_sir/image1.jpeg")      # Loading the image
# image_of_person3 = face_recognition.load_image_file("Known_Persons/Ravitha/ravitha.jpeg")
image_of_person2 = face_recognition.load_image_file("Known_Persons/parvathy/image1.jpeg")
print(image_of_person1.shape)
# print(image_of_person3.shape)

person1_encoding = face_recognition.face_encodings(image_of_person1)[0]
person2_encoding = face_recognition.face_encodings(image_of_person2)[0]
# person2_encoding = face_recognition.face_encodings(image_of_person2)[0]
# print(len(person1_encoding))

known_face_encodings = [person1_encoding,person2_encoding]
known_face_names = ["Nithin","Parvathy"]
