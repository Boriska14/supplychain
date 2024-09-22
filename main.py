import io
import cv2
import numpy as np
import pandas as pd
import pdfplumber
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pyke import knowledge_engine
from pyke import krb_traceback

from model import User, Data_points, Data_enterprise, Data_POST_request

# App object
app = FastAPI()
class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Check if the request URL scheme is 'http'
      if request.url.scheme == "http":
            # Replace 'http' with 'https' in the URL
            url = str(request.url).replace("http://", "https://", 1)
            # Raise an HTTPException with status code 301 and the new 'https' URL
       raise HTTPException(status_code=301, headers={"Location": url})
        # Proceed with the next middleware or request handler
     return await call_next(request)
#Rediction des requete

app.add_middleware(HTTPSRedirectMiddleware)
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CLIENT_SECRET_FILE = "/client_secret_.json"

# importing different methods from database
from database import register_user, login, save_enterprise, save_points, save_data


origins = [
            "https://supplychain.graiperf.eu","https://scgreenoptimizer.fr","http://scgreenoptimizer.fr","https://srv475095.hstgr.cloud", "https://api.graiperf.eu", "http://api.graiperf.eu",]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize the PyKE knowledge engine
engine = knowledge_engine.engine(__file__)

# detect shapes in my document


def detect_shapes(image_content: bytes):
    try:
        # Read the image using OpenCV from the content
        image = cv2.imdecode(np.frombuffer(image_content, np.uint8), -1)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # Find contours in the edged image
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Initialize counters for each shape
        rectangle_count = 0

        # Loop over the contours
        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Get the number of vertices
            vertices = len(approx)

            # Draw the contours on the original image
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

            # Perform inferences based on the number of vertices
            if vertices == 4:
                rectangle_count += 1

        # Display the result
        cv2.imshow("Detected Shapes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return {"rectangle": rectangle_count}
    except Exception as e:
        print(f"Error in shape detection: {e}")
        return {"error": str(e)}


@app.post("/detect_shapes", tags=["Analysing shapes"])
async def extract_shapes_from_doc():
    try:
        docs = get_documents()
        results = []

        for doc in docs:
            print(f"Processing document: {doc['name']}")

            try:
                # Skip PDF documents
                if "pdf" in doc["name"].lower():
                    print(f"Skipping PDF document: {doc['name']}")
                    continue
                # Extract images and detect shapes for each document
                data = detect_shapes(doc["content"])

                # Append results to the list
                results.append({"document_name": doc["name"], "shape_data": data})

            except Exception as e:
                print(f"Error processing document {doc['name']}: {e}")

        # Return the results
        return results
    except HTTPException as e:
        return {"error": str(e)}


# Perform inferences on the extracted data using the PyKE knowledge engine
@app.post("/inferences", tags=["Analysing Documents"])
async def inferences():
    docs1 = await extract_data_from_pdf()
    docs2 = await extract_shapes_from_doc()
    conclusions = []

    # Process PDF documents
    for doc_index, doc in enumerate(docs1):
        doc_name = list(doc["groups"].keys())[0]
        table_number = next(iter(doc["groups"][doc_name].keys()))
        table = doc["groups"][doc_name][table_number]

        # Index each element of the table
        elements = []
        for row_idx, row in enumerate(table["data"]):
            headings = table["headings"]  # Access headings from the table
            for col_idx, cell in enumerate(row):
                if col_idx < len(headings):
                    # Index each cell with its corresponding row and column
                    element = {
                        "row": row_idx,
                        "column": col_idx,
                        "value": cell,
                        "heading": headings[col_idx],
                    }
                    elements.append(element)
        # Perform inferences using the indexed elements
        inferences_results = perform_inferences(elements, doc_index, doc_name)
        conclusions.extend(inferences_results)

    # Process shape extraction documents
    for doc_index, doc in enumerate(docs2):
        document_name = doc["document_name"]
        rectangle_count = doc["shape_data"]["rectangle"]
        # Perform inferences using the indexed elements
        inferences_results2 = perform_inferences_on_shapes(
            rectangle_count, document_name, doc_index
        )
        conclusions.extend(inferences_results2)

    return conclusions  # Return all_elements instead of elements


def perform_inferences_on_shapes(rectangle_count, document_name, doc_index):
    inference_result = {
        "Analysis": "performing good",
        "Data": rectangle_count,
        "DocName": document_name,
        "Document_index": doc_index,
    }
    return [inference_result]


# Function to perform inferences using the indexed elements
def perform_inferences(elements, doc_index, doc_name):
    inferences_result = []
    result_status = None
    inference_result = None
    engine.reset()
    engine.activate("rules")

    try:
        # Calculate the row count dynamically based on the provided elements
        row_count = max(element["row"] + 1 for element in elements)
        # Debugging: Print the calculated row count
        print(f"Calculated Row Count: {row_count}")
        # check for the number of rows and takes a conclusion

        if row_count >= 6:
            with engine.prove_goal("rules.eligible($ans)") as gen:
                for vars, plan in gen:
                    print(f"Rule applied: %s" % (vars["ans"]))
                    result_status = vars["ans"]
                    print(f"result from status: {result_status}")
                    inference_result = {
                        "Analysis": result_status,
                        "DocumentIndex": doc_index,
                        "DocName": doc_name,
                    }
                    inferences_result.append(inference_result)

        # Perform inferences for elements with value=None
        for element in elements:
            if element["value"] == "none":
                with engine.prove_goal("rules.empty($ans)") as gen:
                    for vars, plan in gen:
                        print(f"Rule applied: %s" % (vars["ans"]))
                        result_status = vars["ans"]
                        print(f"Result from status: {result_status}")
                        inference_result = {
                            "Analysis": result_status,
                            "Element": element,
                            "DocumentIndex": doc_index,
                        }
                        inferences_result.append(inference_result)

    except Exception as e:
        # Debugging: Print any exceptions that occur
        print(f"Exception: {e}")
        krb_traceback.print_exc()

    # Debugging: Print the final inference_result
    # print(f"Final Inference Result: {inferences_result}")

    return inferences_result


# End point of the extracted data
@app.post("/load_data", tags=["Analysing Documents"])
async def extract_data_from_pdf():
    # Get the documents
    docs = get_documents()
    file_data = []

    for i, doc in enumerate(docs):
        pdf_content = doc.get("content", None)
        doc_name = doc.get("name", None)

        # Skip JPG documents
        if doc_name.lower().endswith(".jpg"):
            print(f"Skipping JPG document: {doc_name}")
            continue

        if pdf_content is not None:
            data = file_parse(pdf_content, doc_name, i)
            file_data.append(data)

    return file_data


# Function to parse elements from a PDF document
def file_parse(pdf_content, document_name, file_number):
    data = {"groups": {}}

    # Open the PDF using pdfplumber
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf_document:
        for page_num in range(len(pdf_document.pages)):
            page = pdf_document.pages[page_num]
            page_text = page.extract_text()

            # Remove all white spaces from the page text
            cleaned_text = "".join(page_text.split())

            # Extract tables from the page using pdfplumber
            tables = page.extract_tables()

            if tables:
                for i, table in enumerate(tables):
                    if not any(row for row in table):
                        # Skip empty tables
                        continue

                    if document_name not in data["groups"]:
                        data["groups"][document_name] = {}

                    table_number = f"table_{i + 1}"
                    data["groups"][document_name][
                        table_number
                    ] = table_to_tabular_format(table)
            else:
                # If there are no tables on this page, append the cleaned_text to the existing data
                if "text" not in data:
                    data["text"] = ""
                data["text"] += cleaned_text

    # Additional data cleanup before JSON encoding
    cleaned_data = handle_float_values(data)

    # Remove empty strings from the cleaned_data arrays
    cleaned_data = remove_empty_strings(cleaned_data)

    return cleaned_data


# Function to remove empty strings from lists and dictionaries
def remove_empty_strings(data):
    if isinstance(data, list):
        return [item for item in data if item != "" and item != "none"]
    if isinstance(data, dict):
        return {key: remove_empty_strings(value) for key, value in data.items()}
    return data


def table_to_tabular_format(table):
    if table and any(row for row in table):
        headings = table[0]
        data = table[1:]

        # Remove newline characters from the headings
        headings = [head if head is not None else "" for head in headings]
        headings = [head.replace("\n", " ") for head in headings]

        # Remove newline characters from the data
        cleaned_data = []
        for row in data:
            cleaned_row = []
            for cell in row:
                cleaned_cell = cell if cell is not None else ""
                cleaned_cell = cleaned_cell.replace("\n", " ")
                cleaned_row.append(cleaned_cell)
            cleaned_data.append(cleaned_row)

        return {"headings": headings, "data": cleaned_data}
    return {"headings": [], "data": []}


def handle_float_values(data):
    cleaned_data = data.copy()
    for document_name, tables in cleaned_data["groups"].items():
        for table_name, table in tables.items():
            for row in table["data"]:
                if isinstance(
                    row, (list, dict)
                ):  # Check if row is a list or dictionary
                    if isinstance(row, dict):
                        # If row is a dictionary, convert it to a list
                        row = list(row.values())

                    for i in range(len(row)):
                        if isinstance(row[i], float):
                            if pd.isna(row[i]) or pd.isnull(row[i]):
                                row[
                                    i
                                ] = None  # Convert NaN values to None (JSON-compatible null)
                            elif not (-1e308 <= row[i] <= 1e308):
                                # Convert problematic float values to JSON-compatible strings
                                row[i] = f"{row[i]:.10e}"
                        elif row[i] is None or row[i] == "":
                            row[i] = "none"  # Replace blank fields with "none"
    return cleaned_data


# Function to get documents from Google Drive
def get_documents():
    creds = get_google_drive_credentials()
    drive_service = build("drive", "v3", credentials=creds)
    document_criteria = ["pdf", "jpg"]
    query = " or ".join(
        [f"name contains '{criteria}'" for criteria in document_criteria]
    )
    results = (
        drive_service.files()
        .list(pageSize=10, fields="nextPageToken, files(id, name, mimeType)", q=query)
        .execute()
    )
    files = results.get("files", [])

    if not files:
        raise HTTPException(status_code=404, detail="No PDF or JPG files found.")

    documents = []
    for file in files:
        file_id = file["id"]
        file_name = file["name"]
        mime_type = file.get("mimeType", "")

        if "pdf" in mime_type:
            content = get_pdf_content(drive_service, file_id)
        elif "image/jpeg" in mime_type:
            content = get_image_content(drive_service, file_id)
        else:
            # Handle other file types if needed
            content = None

        documents.append(
            {
                "id": file_id,
                "name": file_name,
                "content": content,
            }
        )

    return documents


# Function to get PDF content for a specific document
def get_pdf_content(drive_service, file_id):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    pdf_content = fh.getvalue()
    return pdf_content


# Function to get JPG image content for a specific document
def get_image_content(drive_service, file_id):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    image_content = fh.getvalue()
    return image_content


# Define the '/get_documents' endpoint to retrieve documents
@app.get("/get_documents", tags=["Analysing Documents"])
async def get_documents_endpoint():
    documents = get_documents()
    return {"document": documents}


# Get Google Drive authentication credentials
def get_google_drive_credentials():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
    creds = flow.run_local_server(port=0)
    return creds


@app.get("/")
async def read_root():
    return {"Hey": "Navigate to another page"}


# Register a user in database
@app.post("/register", tags=["USER"])
async def register_users(user: User):
    response = await register_user(user)
    return response


# login a user in database
@app.post("/login", tags=["USER"])
async def login_users(user: User):
    username = user.username
    password = user.password
    response = await login(username, password)
    return response


# save enterprises points into the database
@app.post("/save-points", tags=["DATA"])
async def save_data_points(data: Data_points):
    response = await save_points(data)
    return response


# save enterprises
@app.post("/save-enterprises", tags=["DATA"])
async def save_enterprises(data: Data_enterprise):
    response = await save_enterprise(data)
    return response


# save answers/questions in the database
@app.post("/save-answers", tags=["DATA"])
async def save_answers(data: list[Data_POST_request]):
    response = await save_data(data)
    return response
