import cv2
import imutils
import numpy as np
import pytesseract

"""Page segmentation modes pytesseract:
  0    Orientation and script detection (OSD) only.
  1    Automatic page segmentation with OSD.
  2    Automatic page segmentation, but no OSD, or OCR.
  3    Fully automatic page segmentation, but no OSD. (Default)
  4    Assume a single column of text of variable sizes.
  5    Assume a single uniform block of vertically aligned text.
  6    Assume a single uniform block of text.
  7    Treat the image as a single text line.
  8    Treat the image as a single word.
  9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line,
                        bypassing hacks that are Tesseract-specific. 
                        
Figure source http://www.olavsplates.com/other_countries_submissions.html"""


img = cv2.imread('Car2.jpg', cv2.IMREAD_COLOR) # Load image

resized = cv2.resize(img, (620,480) , interpolation = cv2.INTER_AREA) # Resize image

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) # Convert to grey scale

Blurred = cv2.GaussianBlur(gray,(3,3),2,2) #Remove noise with blurr filter

Edges = cv2.Canny(gray, 60, 150) # Perform Edge detection

IMG_Contours = cv2.findContours(Edges.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Find contours
contours = imutils.grab_contours(IMG_Contours) # Get contours
S_contours = sorted(contours, key = cv2.contourArea, reverse = True) # Sort contours by importance
#cv2.drawContours(resized, S_contours, -1, (0, 255, 0), 3) #Draw contours

for c in S_contours:
    # Approximate contours by series of short line segments
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    # Check the number of points returned
    if len(approx) == 4:
        Plate = approx
        print ("Contour detected")
        cv2.drawContours(resized, [Plate], -1, (0, 0, 255), 3)
        break
    else: 
        Plate = None 

# Deinfe B/W mask
mask = np.zeros(gray.shape, np.uint8)

# Mask the figure using contours
Masked = cv2.drawContours(mask, [Plate], 0, 255, -1,)
Masked = cv2.bitwise_and(resized, resized, mask=mask)

# Find the coordinates of the white square of the mask
(x, y) = np.where(mask == 255)
(upperx, uppery) = (np.min(x), np.min(y))
(lowerx, lowery) = (np.max(x), np.max(y))

# Crop the grayscale figure
Cropped = gray[upperx:lowerx+1, uppery:lowery+1]

# Define Tesseract path 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Identify text using psm 7 
text = pytesseract.image_to_string(Cropped, config='--psm 7')
print('License Plate: ', text)

# Show all figures
cv2.imshow('image',resized)
cv2.imshow('Blurred',Blurred)
cv2.imshow('Edges',Edges)
cv2.imshow('Masked',Masked)
cv2.imshow('Cropped',Cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

