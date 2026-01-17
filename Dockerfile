# Use a Python image with CMake installed
FROM python:3.10-slim

# 1. Install Build Tools (CMake, G++, Make)
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Copy your source code
COPY . /app

# 4. Compile the C++ Extension
RUN mkdir build && cd build && \
    cmake .. && \
    make && \
    mv *.so .. 

# 5. Install Python dependencies (Gradio instead of Tkinter)
RUN pip install numpy gradio pillow

# 6. Run the App
CMD ["python", "app.py"]