import cv2
import os
import csv
import numpy as np
from datetime import datetime

def save_person_image(frame, startX, startY, endX, endY):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")

    directory = os.path.join("persons", year, month, day)
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, f"person_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Persona salvata come {filename}")

    face = frame[startY:endY, startX:endX]
    face_directory = os.path.join("faces", year, month, day)
    if not os.path.exists(face_directory):
        os.makedirs(face_directory)

    face_filename = os.path.join(face_directory, f"face_{timestamp}.jpg")
    cv2.imwrite(face_filename, face)
    print(f"Volto salvato come {face_filename}")

    return os.path.relpath(filename), os.path.relpath(face_filename), now.strftime("%m/%d/%Y"), now.strftime("%H:%M:%S")

def generate_html_report(rows):
    html = """
    <html>
    <head>
        <style>
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        th, td {{
            text-align: left;
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }}
        </style>
    </head>
    <body>
        <h2>Frame salvati</h2>
        <table>
            <tr>
                <th>Link al frame</th>
                <th>Link al volto</th>
                <th>Data</th>
                <th>Ora</th>
            </tr>
            {}
        </table>
    </body>
    </html>
    """

    table_rows = ""
    for row in rows:
        frame_path = os.path.relpath(row[0])
        face_path = os.path.relpath(row[1])
        table_rows += f"<tr><td><a href='{frame_path}'><img width=400px src='{frame_path}'</a></td><td><a href='{face_path}'><img width=400px src='{face_path}'></a></td><td>{row[2]}</td><td>{row[3]}</td></tr>"

    return html.format(table_rows)

def main():
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

    cap = cv2.VideoCapture(0)

    saved_frames = []

    while True:
        ret, frame = cap.read()

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                filename, face_filename, date, time = save_person_image(frame, startX, startY, endX, endY)
                #print('filename percorso:   '+filename)
                filename = ('../'+filename)
                #print('face_filename percorso:   '+face_filename)
                face_filename = ('../'+face_filename)
                
                saved_frames.append([filename, face_filename, date, time])

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    report_dir = "report"

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    html_report = generate_html_report(saved_frames)
    html_filename = os.path.join(report_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    with open(html_filename, "w") as html_file:
        html_file.write(html_report)
    print(f"Report HTML salvato come {os.path.relpath(html_filename)}")

    csv_filename = os.path.join(report_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_filename, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["Frame", "Volto", "Data", "Ora"])
        writer.writerows(saved_frames)
    print(f"Report CSV salvato come {os.path.relpath(csv_filename)}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if not os.path.exists("persons"):
        os.makedirs("persons")
    if not os.path.exists("faces"):
        os.makedirs("faces")

    main()
