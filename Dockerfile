# เลือก base image ของ Python
FROM python:3.10

# สร้าง directory สำหรับงานใน container
WORKDIR /

# คัดลอกไฟล์ requirements.txt ไปยัง container
COPY requirements.txt .

# ติดตั้ง dependencies ใน container
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดทั้งหมดไปยัง container
COPY . .

# เปิดพอร์ต 8000
EXPOSE 8000

# คำสั่งรันแอปพลิเคชัน Django
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
