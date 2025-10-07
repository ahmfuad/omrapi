import cv2
import numpy as np
from imutils import contours
import os
import csv
import logging
import base64
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import io
from PIL import Image
import psycopg2
from psycopg2.extras import RealDictCursor
import json

# --- 1. Setup & Configuration ---

# Flask application setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Database Configuration (UPDATE PASSWORD!) ---
DB_CONFIG = {
    'host': 'aws-1-ap-southeast-1.pooler.supabase.com',
    'database': 'postgres',
    'user': 'postgres.wegyhowolcfcrkmigwuo',
    'password': 'ExamPortalDatabase25', # Your password - UPDATE THIS!
    'port': 5432
}
EXAM_ID = "8ae9128a-717f-4d71-b8e5-a150a3f14812"

def get_db_connection():
    """Create and return a database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")
        return None

def setup_logging():
    """Configures the logging system to output to both a file and the console."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("omr_grader.log"),
                logging.StreamHandler()
            ]
        )

# Initialize logging
setup_logging()

# --- Choice Mapping ---
CHOICE_MAP = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
REVERSE_CHOICE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3}


def get_student_data(roll_number, exam_id=None):
    """
    Fetch student data and answer key from PostgreSQL database
    Returns: dict with student info and answer key or None if not found
    """
    if not psycopg2:
        logging.error("psycopg2 not installed")
        return None
    
    if exam_id is None:
        exam_id = EXAM_ID
    
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return None
            
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
        SELECT
            s.roll_number AS roll,
            s.name AS student_name,
            COALESCE(s.school, 'N/A') AS school_name,
            ec.class_name,
            ec.answer_key
        FROM students s
        JOIN exam_class ec
            ON ec.class_name = s.class
        WHERE s.roll_number = %s
          AND ec.exam_id = %s
        """
        
        cursor.execute(query, (roll_number, exam_id))
        result = cursor.fetchone()
        
        if result:
            student_data = dict(result)
            return {
                'roll_number': student_data.get('roll'),
                'student_name': student_data.get('student_name'),
                'school_name': student_data.get('school_name', 'N/A'),
                'class_name': student_data.get('class_name'),
                'answer_key': student_data.get('answer_key')
            }
        else:
            logging.warning(f"No student data found for roll number: {roll_number}")
            return None
            
    except Exception as e:
        logging.error(f"Database query failed: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def parse_answer_key(answer_key_json):
    """
    Parse answer key from JSON format to the format expected by the grading algorithm
    Input: {"1": "C", "2": "B", "3": "A", ...}
    Output: {0: 2, 1: 1, 2: 0, ...} (0-indexed with choice indices)
    """
    if not answer_key_json:
        return {}
    
    try:
        if isinstance(answer_key_json, str):
            answer_data = json.loads(answer_key_json)
        else:
            answer_data = answer_key_json
        
        parsed = {}
        for question_num, answer in answer_data.items():
            q_index = int(question_num) - 1
            
            if isinstance(answer, list):
                parsed[q_index] = [REVERSE_CHOICE_MAP[ans] for ans in answer if ans in REVERSE_CHOICE_MAP]
            elif isinstance(answer, str) and answer in REVERSE_CHOICE_MAP:
                parsed[q_index] = REVERSE_CHOICE_MAP[answer]
            else:
                logging.warning(f"Invalid answer format for question {question_num}: {answer}")
        
        return parsed
        
    except Exception as e:
        logging.error(f"Failed to parse answer key: {str(e)}")
        return {}

def get_fallback_answer_key():
    """
    Fallback answer key in case database is unavailable
    """
    return {
        0: 1, 1: 2, 2: 0, 3: 2, 4: 1, 5: 0, 6: 1, 7: 2, 8: 3, 9: 0, 10: 1, 11: 3, 12: 0, 13: 2, 14: 1,
        15: 0, 16: 3, 17: 2, 18: 1, 19: 0, 20: 1, 21: 2, 22: [0, 2], 23: 3, 24: 1, 25: 0, 26: 2, 27: 3, 28: 1, 29: 0,
        30: 3, 31: 2, 32: 1, 33: 0, 34: 1, 35: 2, 36: 0, 37: 3, 38: 1, 39: 2, 40: 0, 41: 1, 42: 3, 43: 2, 44: 0,
        45: 1, 46: 3, 47: 0, 48: 2, 49: 1, 50: 3, 51: 0, 52: 1, 53: 2, 54: 3, 55: 0, 56: 1, 57: 2, 58: 3, 59: 0
    }

# --- 2. Core Image Processing Utilities ---


def compress_image_on_the_fly(cv2_image, max_size=1920, quality=75):
    """
    [NEW IN-MEMORY COMPRESSION]
    Converts a cv2 (NumPy) image to a Pillow image, applies resizing and JPEG 
    compression in memory, and converts it back to a cv2 image.

    :param cv2_image: Input image as a NumPy array (BGR format).
    :param max_size: The maximum dimension (longest side) in pixels.
    :param quality: JPEG quality (1-95).
    :return: Compressed image as a NumPy array (BGR format).
    """
    try:
        # 1. Convert cv2 (BGR) to Pillow (RGB)
        # OpenCV reads as BGR, Pillow expects RGB
        img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        
        # 2. Resizing logic
        width, height = img.size
        
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 3. Apply JPEG compression in memory
        # We need a BytesIO object to simulate saving to disk
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, "JPEG", optimize=True, quality=quality)
        buffer.seek(0)
        
        # 4. Read the compressed bytes back into a cv2 image
        file_bytes = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
        compressed_cv2_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        logging.info(f"Image compressed in memory. Original: {width}x{height}, New: {img.size}. Quality: {quality}")
        
        return compressed_cv2_image

    except Exception as e:
        logging.error(f"In-memory compression failed. Reason: {e}. Returning original image.")
        return cv2_image # Fallback: return the original image if compression fails



def reorder(myPoints):
    """Reorders four corner points: top-left, top-right, bottom-left, bottom-right."""
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 2), np.float32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def four_point_transform(image, pts):
    """Applies a perspective transform to an image based on four points."""
    (tl, tr, bl, br) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [0, maxHeight - 1], [maxWidth - 1, maxHeight - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

def find_fiducial_markers(image):
    """Finds the four corner markers on the OMR sheet for alignment."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    marker_contours = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(c)
        if 0.8 <= aspect_ratio <= 1.2 and 100 < area < 10000:
            marker_contours.append(c)
            
    logging.info(f"Found {len(marker_contours)} potential fiducial markers.")
    if len(marker_contours) < 4:
        logging.error("Could not find 4 fiducial markers. Alignment failed.")
        return None
        
    marker_contours = sorted(marker_contours, key=cv2.contourArea, reverse=True)[:4]
    points = []
    for c in marker_contours:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else int(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] / 2)
        cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else int(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] / 2)
        points.append([cX, cY])
        
    return reorder(np.array(points, dtype="float32"))

# --- 3. UNIFIED Bubble Detection Function ---

def find_and_denoise_bubbles(image_section, filter_params):
    """
    A unified function to find bubbles in an image section.
    It uses bilateral filtering for robust, edge-preserving denoising.
    """
    gray = cv2.cvtColor(image_section, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 19, 5)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubble_contours = []
    
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if (filter_params['min_w'] < w < filter_params['max_w'] and
            filter_params['min_h'] < h < filter_params['max_h'] and
            filter_params['min_ar'] <= ar <= filter_params['max_ar']):
            bubble_contours.append(c)
            
    return bubble_contours, gray

# --- 4. Section-Specific Grading Logic ---

def decode_roll_number(roll_section_image):
    """Decodes the roll number from its specific section."""
    if roll_section_image is None or roll_section_image.size == 0:
        return "N/A", 0

    h, w = roll_section_image.shape[:2]
    crop_y_start = int(h * 0.35)
    bubble_area = roll_section_image[crop_y_start:, :]
    
    roll_filter_params = {'min_w': 15, 'max_w': 55, 'min_h': 15, 'max_h': 55, 'min_ar': 0.7, 'max_ar': 1.3}
    bubble_contours, gray = find_and_denoise_bubbles(bubble_area, roll_filter_params)
    
    logging.info(f"Roll Number: Found {len(bubble_contours)} bubbles after filtering.")
    if len(bubble_contours) < 45:
        return f"Not Enough Bubbles ({len(bubble_contours)})", len(bubble_contours)

    bubble_contours, _ = contours.sort_contours(bubble_contours, method="left-to-right")
    columns = []
    current_column = [bubble_contours[0]]
    avg_bubble_width = np.mean([cv2.boundingRect(c)[2] for c in bubble_contours])
    
    for i in range(1, len(bubble_contours)):
        prev_x, _, prev_w, _ = cv2.boundingRect(bubble_contours[i-1])
        curr_x, _, _, _ = cv2.boundingRect(bubble_contours[i])
        if (curr_x - (prev_x + prev_w)) > (avg_bubble_width * 0.4):
            columns.append(current_column)
            current_column = [bubble_contours[i]]
        else:
            current_column.append(bubble_contours[i])
    columns.append(current_column)

    logging.info(f"Clustered roll number bubbles into {len(columns)} columns.")
    
    if 5 <= len(columns) <= 6:
        bubble_columns = columns[-5:]
    else:
        logging.error(f"Expected 5 or 6 columns for roll number, found {len(columns)}.")
        return "Column Count Error", len(bubble_contours)

    decoded_roll = []
    all_intensities = [cv2.mean(gray, mask=cv2.drawContours(np.zeros_like(gray), [c], -1, 255, -1))[0] for c in bubble_contours]
    grading_threshold = np.mean(all_intensities) - (np.std(all_intensities))
    logging.info(f"Roll number grading threshold: {grading_threshold:.2f}")

    for column in bubble_columns:
        column_sorted, _ = contours.sort_contours(column, method="top-to-bottom")
        
        if len(column_sorted) > 10:
            logging.info(f"Found {len(column_sorted)} contours in a column, taking the bottom 10.")
            column_sorted = column_sorted[-10:]

        if len(column_sorted) != 10:
            logging.warning(f"Column has {len(column_sorted)} bubbles after cleanup (expected 10). Skipping.")
            decoded_roll.append("X")
            continue
        
        marked_count = 0
        marked_digit = -1
        for i, bubble_c in enumerate(column_sorted):
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [bubble_c], -1, 255, -1)
            if cv2.mean(gray, mask=mask)[0] < grading_threshold:
                marked_digit = i
                marked_count += 1
        
        decoded_roll.append(str(marked_digit) if marked_count == 1 else "E")
        
    return "".join(decoded_roll), len(bubble_contours)


def grade_mcq_section(section_image, question_offset, output_image, section_x_offset, section_y_offset, answer_key):
    """Grades a single MCQ section using a robust, per-question grading method."""
    if section_image is None or section_image.size == 0:
        return [], 0, 0

    h_sec, w_sec = section_image.shape[:2]
    y_crop_offset = int(h_sec * 0.05)
    bubble_area = section_image[y_crop_offset:, :]
    
    mcq_filter_params = {'min_w': 8, 'max_w': 60, 'min_h': 8, 'max_h': 60, 'min_ar': 0.2, 'max_ar': 5.0}
    bubble_contours, gray = find_and_denoise_bubbles(bubble_area, mcq_filter_params)
    
    logging.info(f"MCQ Section {question_offset//15+1}: Found {len(bubble_contours)} bubbles.")
    if not bubble_contours:
        return [], 0, 0

    question_rows = []
    if bubble_contours:
        bubble_contours = contours.sort_contours(bubble_contours, method="top-to-bottom")[0]
        row = [bubble_contours[0]]
        for i in range(1, len(bubble_contours)):
            if abs(cv2.boundingRect(bubble_contours[i])[1] - cv2.boundingRect(row[0])[1]) < 20:
                row.append(bubble_contours[i])
            else:
                question_rows.append(row)
                row = [bubble_contours[i]]
        question_rows.append(row)

    section_answers = []
    section_correct = 0

    for row_idx, row in enumerate(question_rows):
        question_num = question_offset + row_idx
        
        if len(row) < 4:
            logging.warning(f"Q:{question_num+1} - Found only {len(row)} bubbles. Skipping.")
            continue
        if len(row) > 5:
            logging.warning(f"Q:{question_num+1} - Found {len(row)} bubbles, likely noise. Taking rightmost 4.")
            row = contours.sort_contours(row, method="left-to-right")[0][-4:]
        if len(row) == 5:
            row = contours.sort_contours(row, method="left-to-right")[0][1:]
        
        question_bubbles = contours.sort_contours(row, method="left-to-right")[0]
        
        intensities = []
        for bubble_contour in question_bubbles:
            mask = np.zeros_like(gray)
            (x, y, w, h) = cv2.boundingRect(bubble_contour)
            center_x, center_y = x + w // 2, y + h // 2
            radius = int(min(w, h) / 2 * 0.707) 
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            intensities.append(cv2.mean(gray, mask=mask)[0])

        marked_choice_idx = -1
        marked_count = 0
        
        if len(intensities) == 4:
            sorted_intensities = sorted(intensities)
            min_intensity = sorted_intensities[0]
            second_min_intensity = sorted_intensities[1]
            max_intensity = sorted_intensities[3]

            CONTRAST_THRESHOLD = 60 
            BLANK_THRESHOLD = 170
            SEPARATION_GAP = 20

            if (max_intensity - min_intensity) > CONTRAST_THRESHOLD and min_intensity < BLANK_THRESHOLD:
                if (second_min_intensity - min_intensity) > SEPARATION_GAP:
                    marked_count = 1
                    marked_choice_idx = intensities.index(min_intensity)
                else:
                    dark_threshold = second_min_intensity + (SEPARATION_GAP / 2.0)
                    dark_bubbles = [i for i in intensities if i < dark_threshold]
                    marked_count = len(dark_bubbles)
            else:
                marked_count = 0
        
        is_correct = False
        correct_answers = answer_key.get(question_num)

        if marked_count == 1:
            if correct_answers is None:
                is_correct = False
            elif isinstance(correct_answers, list):
                if marked_choice_idx in correct_answers:
                    is_correct = True
            else:
                if marked_choice_idx == correct_answers:
                    is_correct = True
        
        if is_correct:
            section_correct += 1
            
        student_ans_char = CHOICE_MAP.get(marked_choice_idx, 'Blank') if marked_count == 1 else ('Error' if marked_count > 1 else 'Blank')
        correct_ans_char = ", ".join([CHOICE_MAP.get(ans) for ans in correct_answers]) if isinstance(correct_answers, list) else CHOICE_MAP.get(correct_answers, 'N/A')
        
        section_answers.append({'question': question_num + 1, 'student_answer': student_ans_char, 'correct_answer': correct_ans_char, 'result': 'Correct' if is_correct else 'Incorrect'})
        
        if correct_answers is not None:
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            answers_to_draw = correct_answers if isinstance(correct_answers, list) else [correct_answers]
            
            for answer_idx in answers_to_draw:
                if answer_idx < len(question_bubbles):
                    (x_b, y_b, w_b, h_b) = cv2.boundingRect(question_bubbles[answer_idx])
                    final_x = x_b + section_x_offset
                    final_y = y_b + section_y_offset + y_crop_offset
                    cv2.rectangle(output_image, (final_x, final_y), (final_x + w_b, final_y + h_b), color, 3)
                    
            if not is_correct and marked_choice_idx != -1 and marked_count == 1:
                (x_b, y_b, w_b, h_b) = cv2.boundingRect(question_bubbles[marked_choice_idx])
                final_x = x_b + section_x_offset
                final_y = y_b + section_y_offset + y_crop_offset
                cv2.circle(output_image, (final_x + w_b//2, final_y + h_b//2), int(w_b/2), (0, 0, 255), 2)
                
    return section_answers, section_correct, len(bubble_contours)

# --- 5. Main Processing Pipeline ---

def process_omr_sheet(image_path):
    """Main function to load an image and run the entire grading process."""
    original_image = cv2.imread(image_path)
    if original_image is None:
        logging.error(f"Could not load image from {image_path}. Check the path.")
        return {
            'success': False,
            'message': 'Could not load image.'
        }

    compressed_image = compress_image_on_the_fly(original_image, max_size=1920, quality=75)


    # 1. Find and align the sheet
    corners = find_fiducial_markers(compressed_image)
    fiducial_markers_found = corners is not None
    if not fiducial_markers_found:
        return {
            'success': False,
            'message': 'Could not find 4 fiducial markers.'
        }
    
    paper = four_point_transform(compressed_image, corners)
    h_paper, w_paper = paper.shape[:2]
    crop_margin_top = int(h_paper * 0.02)
    crop_margin_bottom = int(h_paper * 0.02)
    crop_margin_x_left = int(w_paper * 0.04)
    crop_margin_x_right = int(w_paper * 0.02)
    cropped_paper = paper[crop_margin_top:h_paper - crop_margin_bottom, crop_margin_x_left:w_paper - crop_margin_x_right]
    output_image = cropped_paper.copy()
    
    h_crop, w_crop, _ = cropped_paper.shape
    split_y = int(h_crop * 0.22)
    gray_split = cv2.cvtColor(cropped_paper, cv2.COLOR_BGR2GRAY)
    thresh_split = cv2.adaptiveThreshold(cv2.GaussianBlur(gray_split, (5,5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    cnts_split, _ = cv2.findContours(thresh_split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_split:
        x, y, w, h = cv2.boundingRect(c)
        if w > w_crop * 0.8 and h < 20:
            split_y = y + h // 2
            logging.info(f"Found horizontal separator at y={split_y}")
            break
            
    roll_section = cropped_paper[0:split_y, :]
    mcq_section_full = cropped_paper[split_y:, :]

    # 2. Decode the roll number
    roll_number, roll_bubbles = decode_roll_number(roll_section)
    logging.info(f"--- Decoded Roll Number: {roll_number} ---")
    roll_number_decoded = roll_number.isdigit()

    # 3. Get student data and answer key from database or use fallback
    student_data = None
    using_fallback_answer_key = True
    answer_key_to_use = {}

    if roll_number_decoded:
        student_data = get_student_data(roll_number, EXAM_ID)
        if student_data and student_data.get('answer_key'):
            answer_key_to_use = parse_answer_key(student_data['answer_key'])
            using_fallback_answer_key = False
            logging.info(f"Successfully retrieved answer key for roll number {roll_number} from DB.")
        else:
            logging.warning("Database lookup failed or no answer key found. Using fallback.")
            answer_key_to_use = get_fallback_answer_key()
    else:
        logging.warning("Roll number not decoded correctly. Using fallback answer key.")
        answer_key_to_use = get_fallback_answer_key()

    database_lookup_success = student_data is not None and not using_fallback_answer_key
    student_name = student_data.get('student_name', 'N/A') if student_data else 'N/A'
    school_name = student_data.get('school_name', 'N/A') if student_data else 'N/A'
    class_name = student_data.get('class_name', 'N/A') if student_data else 'N/A' 
    total_questions = len(answer_key_to_use)

    # 4. Grade the MCQ sections
    total_correct = 0
    all_student_answers = []
    total_mcq_bubbles = 0
    
    mcq_gray = cv2.cvtColor(mcq_section_full, cv2.COLOR_BGR2GRAY)
    mcq_thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(mcq_gray, (5,5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_lines = cv2.morphologyEx(mcq_thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    cnts_lines, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    separator_xs = []
    for c in cnts_lines:
        if cv2.boundingRect(c)[3] > mcq_section_full.shape[0] * 0.5:
            separator_xs.append(cv2.boundingRect(c)[0])
    
    separator_xs.sort()
    
    vertical_separators_found = len(separator_xs) >= 3
    if vertical_separators_found:
        logging.info(f"Found vertical separators at x={separator_xs}")
        split_x1, split_x2, split_x3 = separator_xs[0], separator_xs[1], separator_xs[2]

        sections = [
            (mcq_section_full[:, :split_x1], 0, 0),
            (mcq_section_full[:, split_x1:split_x2], 15, split_x1),
            (mcq_section_full[:, split_x2:split_x3], 30, split_x2),
            (mcq_section_full[:, split_x3:], 45, split_x3)
        ]

        for sec_img, q_offset, x_offset in sections:
            answers, correct, bubbles_found = grade_mcq_section(sec_img, q_offset, output_image, x_offset, split_y, answer_key_to_use)
            all_student_answers.extend(answers)
            total_correct += correct
            total_mcq_bubbles += bubbles_found
    else:
        logging.error(f"Failed to find 3 clear vertical separators. Found {len(separator_xs)}. Grading aborted.")
        
    score = total_correct
    score_percentage = (total_correct / total_questions) * 100 if total_questions > 0 else 0
    total_bubbles_detected = roll_bubbles + total_mcq_bubbles
    
    logging.info(f"--- OMR Grading Results ---\nCorrect Answers: {total_correct} / {total_questions}\nScore: {score_percentage:.2f}%")
    
    all_student_answers.sort(key=lambda x: x['question'])
    
    # Add score and roll number text to the image
    cv2.putText(output_image, f"Roll: {roll_number}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 100, 0), 3)
    cv2.putText(output_image, f"Score: {score:.2f}%", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    # cv2.putText(output_image, f"Name: {student_name}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    # cv2.putText(output_image, f"School: {school_name}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)


    # Encode the processed image to Base64
    _, buffer = cv2.imencode('.jpg', output_image)
    processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # --- Response Structure ---
    response = {
        'success': True,
        'roll_number': roll_number,
        'score': score,
        'score_percentage': round(score_percentage, 2),
        'total_questions': total_questions,
        'answers': all_student_answers,
        'processed_image': processed_image_base64,
        'processing_details': {
            'fiducial_markers_found': fiducial_markers_found,
            'roll_number_decoded': roll_number_decoded,
            'vertical_separators_found': vertical_separators_found,
            'total_bubbles_detected': total_bubbles_detected,
            'database_lookup_success': database_lookup_success,
            'using_fallback_answer_key': using_fallback_answer_key
        }
    }
    
    #  Add student information if available
    if student_data:
        response['student_info'] = {
            'name': student_name,
            'school': school_name,
            'class': class_name
        }
    
    return response

# --- Flask API Endpoint ---

@app.route('/process-omr', methods=['POST'])
def process_omr_api():
    setup_logging()
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'message': 'No file part in the request.'
        }), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({
            'success': False,
            'message': 'No selected file.'
        }), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # Read image from stream to avoid saving to disk first
            image_stream = io.BytesIO(file.read())
            img_pil = Image.open(image_stream)
            img_np = np.array(img_pil)

            if img_np.ndim == 3 and img_np.shape[2] == 3:
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                img_cv = img_np
            
            cv2.imwrite(filepath, img_cv)

            result = process_omr_sheet(filepath)
            
            os.remove(filepath)
            
            return jsonify(result), 200
        
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            return jsonify({
                'success': False,
                'message': f"An error occurred during processing: {str(e)}"
            }), 500

application = app
# --- WSGI entry point ---
if __name__ == '__main__':
    app.run(debug=True)
