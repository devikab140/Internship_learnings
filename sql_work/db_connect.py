import requests

# Define the API base URLs
student_api = "http://192.168.1.35:5000/api/Department"
hospital_api = "http://192.168.1.68:5000/api/beds"

# Get student data
try:
    stu_response = requests.get(student_api)
    stu_response.raise_for_status()  # check for errors
    students = stu_response.json()
    print(" Student Data:")
    print(students)
except Exception as e:
    print("Error fetching student data:", e)

# Get hospital data
try:
    hosp_response = requests.get(hospital_api)
    hosp_response.raise_for_status()
    hospital = hosp_response.json()
    print("\n🏥 Hospital Data:")
    print(hospital)
except Exception as e:
    print("Error fetching hospital data:", e)
