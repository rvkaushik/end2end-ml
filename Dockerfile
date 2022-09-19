FROM python:3.8-slim
COPY . /end2end
WORKDIR /end2end
# RUN pip install --upgrade pip
RUN pip install --upgrade pip --no-cache-dir -r requirement.txt
ENTRYPOINT ["python3",  "infer.py"]
