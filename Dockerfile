FROM python
COPY . /end2end
WORKDIR /end2end
RUN pip install --upgrade pip
RUN pip install -r requirement.txt
ENTRYPOINT ["python3",  "infer.py"]
