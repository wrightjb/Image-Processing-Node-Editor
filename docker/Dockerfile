FROM python:3.13.2-alpine3.21@sha256:323a717dc4a010fee21e3f1aac738ee10bb485de4e7593ce242b36ee48d6b352

# Install system libs (runtime only, not dev headers)
RUN apk add --no-cache \
    mesa-gl \
    glfw \
    freetype \
    libx11 \
    libxext \
    libxrender \
    libxrandr \
    libxcursor \
    libxi \
    libxinerama \
    ffmpeg \
    gcompat \
    libstdc++ \
    py3-pip

# Copy wheels into image
COPY wheels/ /wheels/

# Install from local wheels only
RUN pip install --no-cache-dir --find-links=/wheels --no-index \
    opencv-contrib-python \
    dearpygui \
    pytest

# Set working directory and copy project files
WORKDIR /app
COPY . .

CMD ["sh"]