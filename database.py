import sqlite3
import pickle

DATABASE_PATH = 'face_recognition.db'

def create_connection():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
    except sqlite3.Error as e:
        print(e)
    return conn

def create_table():
    conn = create_connection()
    if conn is not None:
        try:
            conn.execute('''CREATE TABLE IF NOT EXISTS registered_faces
                         (student_id TEXT PRIMARY KEY,
                          face_embedding BLOB,
                          additional_info TEXT)''')
        except sqlite3.Error as e:
            print(e)
        finally:
            conn.close()

def insert_face_data(student_id, face_embedding, additional_info=''):
    conn = create_connection()
    if conn is not None:
        serialized_embedding = pickle.dumps(face_embedding)
        try:
            conn.execute("INSERT INTO registered_faces (student_id, face_embedding, additional_info) VALUES (?, ?, ?)",
                         (student_id, serialized_embedding, additional_info))
            conn.commit()
        except sqlite3.Error as e:
            print(e)
        finally:
            conn.close()

def get_face_data(student_id):
    conn = create_connection()
    face_data = None
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM registered_faces WHERE student_id = ?", (student_id,))
            data = cursor.fetchone()
            if data:
                face_embedding = pickle.loads(data[1])
                face_data = (face_embedding, data[2])
        except sqlite3.Error as e:
            print(e)
        finally:
            conn.close()
    return face_data

def get_all_student_ids():
    conn = create_connection()
    student_ids = []
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT student_id FROM registered_faces")
            rows = cursor.fetchall()
            for row in rows:
                student_ids.append(row[0])
        except sqlite3.Error as e:
            print(e)
        finally:
            conn.close()
    return student_ids

create_table()
