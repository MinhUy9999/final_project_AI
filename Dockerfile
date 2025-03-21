# Sử dụng image Python chính thức làm base
FROM python:3.8-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép requirements.txt và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Đảm bảo thư mục model_weights tồn tại (nếu cần tải mô hình trong runtime)
RUN mkdir -p model_weights

# Cài đặt Gunicorn để chạy Flask
RUN pip install gunicorn

# Mở cổng 5000 (hoặc cổng Render yêu cầu)
EXPOSE 5000

# Chạy ứng dụng bằng Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]