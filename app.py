from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import cv2
import numpy as np
from joblib import load
import pickle
from datetime import datetime

app = Flask(__name__)

def process_image(image_path):
    # Your provided code here...
    # ...
    img0 = cv2.imread(image_path)
    resized_image1 = cv2.resize(img0, (800, 600))
    resized_image = img0.copy()
    width1, height1, _ = resized_image.shape
    dim_org = [width1, height1]

    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(gray_image, (7, 7), sigmaX=0, sigmaY=0)

    lower = 0  # Example lower intensity threshold
    upper = 81  # Example upper intensity threshold

    # Apply a binary threshold to get a mask of pixels within the specified range
    mask = cv2.inRange(img_blur, lower, upper)

    # Convert the mask to a binary image using cv2.threshold
    ret, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    #ret, thresh = cv2.threshold(img_blur, 73, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    # x1, y1, w1, h1 = cv2.boundingRect(cnt)

    # rect = cv2.minAreaRect(cnt)
    # rect_points = cv2.boxPoints(rect)
    # rect_points = np.int0(rect_points)

    max_contour = max(contours, key=cv2.contourArea)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    epsilon = 0.05 * cv2.arcLength(contours_sorted[1], True)
    approx = cv2.approxPolyDP(contours_sorted[1], epsilon, True)

    # epsilon = 0.05 * cv2.arcLength(max_contour, True)
    # approx = cv2.approxPolyDP(max_contour, epsilon, True)

    # min_x = np.min(rect_points[:, 0])
    # max_x = np.max(rect_points[:, 0])
    # min_y = np.min(rect_points[:, 1])
    # max_y = np.max(rect_points[:, 1])

    # # Crop the region of interest (ROI) from the original image
    # cropped_image = resized_image[min_y:max_y, min_x:max_x]
    sorted_coords = sorted(approx, key=lambda x: x[0][0] + x[0][1])

    # Reassign the sorted coordinates
    bottom_left = sorted_coords[0]
    sorted_coords.pop(0)
    top_right = sorted_coords[2]
    sorted_coords.pop(2)

    print(sorted_coords[1][0])

    if sorted_coords[0][0][1] > sorted_coords[1][0][1]:
        selected_coordinate = sorted_coords[0]
        sorted_coords.pop(0)
    else:
        selected_coordinate = sorted_coords[1]
        sorted_coords.pop(1)

    top_left = selected_coordinate
    bottom_right = sorted_coords[0]

    pt_A = bottom_left
    pt_B = top_left
    pt_C = top_right
    pt_D = bottom_right

    # pt_A = approx[0]
    # pt_B = approx[1]
    # pt_C = approx[2]
    # pt_D = approx[3]

    width_AD = np.sqrt(((pt_A[0,0] - pt_D[0,0]) ** 2) + ((pt_A[0,1] - pt_D[0,1]) ** 2))
    width_BC = np.sqrt(((pt_B[0,0] - pt_C[0,0]) ** 2) + ((pt_B[0,1] - pt_C[0,1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0,0] - pt_B[0,0]) ** 2) + ((pt_A[0,1] - pt_B[0,1]) ** 2))
    height_CD = np.sqrt(((pt_C[0,0] - pt_D[0,0]) ** 2) + ((pt_C[0,1] - pt_D[0,1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                            [0, maxHeight - 1],
                            [maxWidth - 1, maxHeight - 1],
                            [maxWidth - 1, 0]])

    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts, output_pts)

    # Apply perspective transformation
    out = cv2.warpPerspective(resized_image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    rotated_image=out

    # out1 = cv2.flip(out, 1)
    # # Convert the result to RGB for display
    # #out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    # rotated_image = cv2.rotate(out1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # rotated_image1 = cv2.rotate(rotated_image , cv2.ROTATE_90_COUNTERCLOCKWISE)

    # if(rect[2]>=49 and rect[2]<=90):
    #     out1 = cv2.flip(out, 1)
    #     rotated_image = cv2.rotate(out1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #     #rotated_image = cv2.flip(rotated_image, 1)
    #     #print("case1")
    # else:
    #     out1 = cv2.flip(out, 1)
    #     rotated_image = cv2.rotate(out1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #     rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #     #print("case2")

    # source = cv2.resize(rotated_image, (1267,1428))
    source = rotated_image
    width, height, _ = source.shape
    dim = [width, height]
    source = cv2.resize(rotated_image, (5412,6142),interpolation=cv2.INTER_CUBIC)

    current_directory = os.getcwd()
    model_filename = "best5.pt"
    model_filepath = os.path.join(current_directory, model_filename)
    model = YOLO(model_filepath)  # pretrained YOLOv8n model 'C:\\Windows\\System32\\runs\\detect\\train5\\weights\\best.pt'
    #source = cv2.imread(image_path)
    results = model(source)


    # Draw the rotated rectangle on the original image
    # img_with_rect = resized_image.copy()  # Make a copy of the original image
    # cv2.polylines(img_with_rect, [rect_points], isClosed=True, color=(0, 255, 0), thickness=2)

    # pt_A = rect_points[0]
    # pt_B = rect_points[1]
    # pt_C = rect_points[2]
    # pt_D = rect_points[3]

    # width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    # width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    # maxWidth = max(int(width_AD), int(width_BC))

    # height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    # height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    # maxHeight = max(int(height_AB), int(height_CD))

    # # Define input and output points
    # input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    # output_pts = np.float32([[0, 0],
    #                         [0, maxHeight - 1],
    #                         [maxWidth - 1, maxHeight - 1],
    #                         [maxWidth - 1, 0]])

    # # Compute the perspective transform M
    # M = cv2.getPerspectiveTransform(input_pts, output_pts)

    # # Apply perspective transformation
    # out = cv2.warpPerspective(resized_image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    # # out1 = cv2.flip(out, 1)
    # # # Convert the result to RGB for display
    # # #out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    # # rotated_image = cv2.rotate(out1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # # rotated_image1 = cv2.rotate(rotated_image , cv2.ROTATE_90_COUNTERCLOCKWISE)

    # if(rect[2]>=49 and rect[2]<=90):
    #     out1 = cv2.flip(out, 1)
    #     rotated_image = cv2.rotate(out1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #     #rotated_image = cv2.flip(rotated_image, 1)
    #     #print("case1")
    # else:
    #     out1 = cv2.flip(out, 1)
    #     rotated_image = cv2.rotate(out1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #     rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #     #print("case2")

    # source = cv2.resize(rotated_image, (1267,1428))
    # width, height, _ = source.shape
    # dim = [width, height]

    # current_directory = os.getcwd()
    # model_filename = "best3.pt"
    # model_filepath = os.path.join(current_directory, model_filename)
    # model = YOLO(model_filepath)  # pretrained YOLOv8n model 'C:\\Windows\\System32\\runs\\detect\\train5\\weights\\best.pt'
    # #source = cv2.imread(image_path)
    # results = model(source) 

    for r in results:
        #  print(r.boxes.xyxy)
        coord_list = r.boxes.xyxy.tolist()
        #print(r.boxes.conf)
        #  print(r.boxes)
        conf_list=r.boxes.conf.tolist()
        #  print(r.boxes)
# print(type(coord_list[0][0]))
# print(len(coord_list))

    image = np.copy(source)

    rice_type=[]
    chalkiness=[]
    pos=[]
    ar=[]
    length=[]
    breadth=[]
    index=[]
    tbas=0
    tnbas=0

    filtered_coords = []

    # Iterate over the coordinates and corresponding confidence scores
    for coords, confidence in zip(coord_list, conf_list):
        if confidence > 0.5:
            filtered_coords.append(coords)

    sorted_sublists = sorted(filtered_coords, key=lambda x: x[3])

    mult1 = 58.86/len(source)#0.0425 61.42
    mult2 = 52.43/len(source[0])#0.0425 54.12 51.43
    # Loop through the bounding boxes

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Loop through the bounding boxes


    for i, (startX, startY, endX, endY) in enumerate(filtered_coords):
    # Extract the region of interest (ROI)

        startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
        roi = image[startY:endY, startX:endX]
        img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (7, 7), sigmaX=0, sigmaY=0)
        ret, thresh = cv2.threshold(img_blur, 60, 255, cv2.THRESH_BINARY) #73

        # Calculate proportion of white pixels
        white_pixels = cv2.countNonZero(thresh)
        total_pixels = thresh.size
        white_proportion = white_pixels / total_pixels

        if white_proportion >= 0.17:  # Adjust this threshold as needed
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Find the index of the largest contour
                areas = [cv2.contourArea(c) for c in contours]
                max_index = np.argmax(areas)
                cnt = contours[max_index]

                #pairwise_distances = [np.linalg.norm(cnt[i] - cnt[j]) for i in range(len(cnt)) for j in range(i + 1, len(cnt))]

                # Find the maximum distance
                #max_distance = max(pairwise_distances)

                M = cv2.moments(cnt)
                # Calculate centroid coordinates
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])

                #num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

                # Find the label of the largest connected component (excluding background)
                #largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

                # Find the length of the largest connected component
                #ln = stats[largest_label, cv2.CC_STAT_WIDTH]


                width = thresh.shape[1]
                # p = 0
                # while (p + centroid_x < width - 1):
                #     if thresh[centroid_y][p + centroid_x] != thresh[centroid_y][p + centroid_x + 1]:
                #         x1_i = p + centroid_x
                #         break  # Stop the loop as soon as intensity change is detected
                #     p += 1

                # # Iterate while intensity changes downwards
                # q = 0
                # while (centroid_x - q >= 0):
                #     if thresh[centroid_y][centroid_x - q] != thresh[centroid_y][centroid_x - q - 1]:
                #         x2_i = centroid_x - q
                #         break  # Stop the loop as soon as intensity change is detected
                #     q += 1






                height = thresh.shape[0]
                y1_i = centroid_y
                y2_i = centroid_y

                k=0
                while(k+centroid_y<height-1):

                    if thresh[k+centroid_y][centroid_x] != thresh[k+centroid_y+1][centroid_x]:
                        y1_i = k+centroid_y
                        break
                    k=k+1

                j=0

                while(centroid_y-j>=0):
                    if thresh[centroid_y-j][centroid_x] != thresh[centroid_y-j-1][centroid_x]:
                        y2_i = centroid_y - j
                        break
                    j=j+1
                #above = (y1_i, centroid_x)
                #below = (y2_i, centroid_x)

                rect = cv2.minAreaRect(cnt)
                rect_points = cv2.boxPoints(rect)
                rect_points = np.intp(rect_points)
                #cv2.polylines(image, [rect_points], isClosed=True, color=(0, 0, 255), thickness=2)
                h1,w1 = rect[1]
                #x1,y1=rect_points[3]

                if h1>=w1:
                    l1 = h1*mult2
                    b1 = w1*mult1

                else:
                    l1 = w1*mult2
                    b1 = h1*mult1
                    h1,w1 = w1,h1


                x1, y1, w2, h2 = cv2.boundingRect(cnt)
                # l1 = w1*mult2
                # b1 = h1*mult1
                # Define the path to the saved model
                #current_directory = os.getcwd()
                # model_filename1 = "decision_tree_model.pkl"
                # model_filepath1 = os.path.join(current_directory, model_filename1)
                # # # #model_path = "decision_tree_model.joblib"  # Adjust the path if it's different

                # # # # Load the saved model
                # loaded_model = load(model_filepath1)


                X_test = np.array([[l1]])  # Example test data
                # predicted_y = loaded_model.predict(X_test)

                b1 = (y1_i-y2_i)*mult1
                #l1 = (max_distance*mult2)

                #l2 = l1-0.26

                length.append(l1) #predicted_y[0]
                breadth.append(b1)


                adjusted_box = rect_points.copy()
                adjusted_box[:, 0] += startX  # Adjust x coordinates
                adjusted_box[:, 1] += startY  # Adjust y coordinates
                # Draw bounding box on original image
                cv2.drawContours(image, [adjusted_box], 0, (0, 255, 0), 10)
                # Draw bounding box on original image
                #cv2.rectangle(image, (startX + x1, startY + y1), (startX + x1 + w1, startY + y1 + h1), (0, 255, 0), 2)
                #cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)
                #cv2.putText(image, str(i), (startX+x1, startY+y1+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                aspect_ratio = float(h1) / w1
                ar.append(aspect_ratio)
                index.append(i)

                ret1, thresh1 = cv2.threshold(img_blur, 142, 255, cv2.THRESH_BINARY) #150

                white1 = cv2.countNonZero(thresh1)
                total1 = thresh1.size
                white_proportion1 = white1 / total1

                if l1<6.61:
                    rice_type.append(f"Non-Basmati_{i}")
                    cv2.putText(image, str(i), (startX+x1, startY+y1+100), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 16) #y=10
                    cv2.putText(image, "NB", (startX+x1+220, startY+y1+100), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 16) #70 (255,0,0)
                    pos.append(filtered_coords[i])
                    tnbas=tnbas+1
                    # cv2.drawContours(tempimg, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)
                    if white_proportion1 >= 0.02:
                        chalkiness.append(f"chalky_{i}")
                        cv2.putText(image, "C", (startX+x1+500, startY+y1+100), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 16) #130
                    else:
                        chalkiness.append(f"non-chalky_{i}")
                        cv2.putText(image, "NC", (startX+x1+500, startY+y1+100), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 16)


                elif b1>2.1:
                    rice_type.append(f"Non-Basmati_{i}")
                    cv2.putText(image, str(i), (startX+x1, startY+y1+100), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 16)
                    cv2.putText(image, "NB", (startX+x1+220, startY+y1+100), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 16)
                    pos.append(filtered_coords[i])
                    tnbas=tnbas+1
                    # cv2.drawContours(tempimg, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)
                    if white_proportion1 >= 0.02:
                        chalkiness.append(f"chalky_{i}")
                        cv2.putText(image, "C", (startX+x1+500, startY+y1+100), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 16)
                    else:
                        chalkiness.append(f"non-chalky_{i}")
                        cv2.putText(image, "NC", (startX+x1+500, startY+y1+100), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 16)

                elif aspect_ratio < 3.5:
                    rice_type.append(f"Non-Basmati_{i}")
                    cv2.putText(image, str(i), (startX+x1, startY+y1+100), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 16)
                    cv2.putText(image, "NB", (startX+x1+220, startY+y1+100), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 16)
                    pos.append(filtered_coords[i])
                    tnbas=tnbas+1
                    # cv2.drawContours(tempimg, [cnt], -1, (0, 0, 255), thickness=cv2.FILLED)
                    if white_proportion1 >= 0.02:
                        chalkiness.append(f"chalky_{i}")
                        cv2.putText(image, "C", (startX+x1+500, startY+y1+100), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 16)
                    else:
                        chalkiness.append(f"non-chalky_{i}")
                        cv2.putText(image, "NC", (startX+x1+500, startY+y1+100), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 16)
                else:
                    rice_type.append(f"Basmati_{i}")
                    cv2.putText(image, str(i), (startX+x1, startY+y1+100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 16)
                    cv2.putText(image, "B", (startX+x1+220, startY+y1+100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 16)
                    pos.append(filtered_coords[i])
                    tbas=tbas+1
                    # cv2.drawContours(tempimg, [cnt], -1, (255, 0, 0), thickness=cv2.FILLED)
                    if white_proportion1 >= 0.02:
                        chalkiness.append(f"chalky_{i}")
                        cv2.putText(image, "C", (startX+x1+500, startY+y1+100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 16)

                    else:
                        chalkiness.append(f"non-chalky_{i}")
                        cv2.putText(image, "NC", (startX+x1+500, startY+y1+100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 16)


            else:
                #print("Not enough white pixels at position")
                pass
    if(len(sorted_sublists)==0):
        average=0
        min_length=0
        return rice_type, chalkiness, pos, ar, image, resized_image1, average, index, tbas, tnbas, min_length, length, breadth, dim, mult1, mult2, timestamp, dim_org

    total = sum(length)
    average = total / len(length)
    min_length = min(len(index), len(rice_type), len(chalkiness), len(ar))
    return rice_type, chalkiness, pos, ar, image, resized_image1, average, index, tbas, tnbas, min_length, length, breadth, dim, mult1, mult2, timestamp, dim_org

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Get the uploaded file
    uploaded_file = request.files['file']

    if uploaded_file.filename != '':
        # Save the uploaded file
        uploaded_file.save('static//uploads//uploaded_image.jpg')

        # Process the uploaded image
        rice_type, chalkiness, pos, ar, source, resized_image1, average, index, tbas, tnbas, min_length, length, breadth, dim, mult1, mult2, timestamp, dim_org  = process_image('static//uploads//uploaded_image.jpg')
        
        # Save the processed image
        cv2.imwrite('static//resized_image1.jpg', resized_image1)
        cv2.imwrite('static//processed_image.jpg', source)  # Assuming 'source' is the processed image

        # Pass the lists to the result template
        return render_template('result.html', 
                               rice_type=rice_type, 
                               chalkiness=chalkiness, 
                               pos=pos, 
                               ar=ar,
                               average=average, 
                               index=index,
                               tbas=tbas,
                               tnbas=tnbas,
                               min_length=min_length, 
                               length=length,
                               breadth=breadth,
                               dim=dim,
                               mult1=mult1,
                               mult2=mult2,
                               timestamp=timestamp,
                               dim_org=dim_org,
                               resized_image1='static//resized_image1.jpg', 
                               processed_image='static//processed_image.jpg')
    else:
        return "No file uploaded"

if __name__ == '__main__':
    app.run(debug=True)