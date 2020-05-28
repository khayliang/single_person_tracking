from deep_sort import DeepSort
import numpy as np


if __name__ == "__main__":
    faces = np.array([
        [40,33,120,150],
        [50,33,120,150],
        [10000,5000,120,150]
        ])

    body = np.array([[50,33,120,150],[500,303,120,150]])
    tuples_list = DeepSort._match_faces(faces,body)
    unzip_tuples = [list(t) for t in zip(*tuples_list)]
    faces_list = np.array(unzip_tuples[0])
    bodies_list = np.array(unzip_tuples[1])
    print(faces_list)