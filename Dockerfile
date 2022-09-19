# Stage 1 - Install build dependencies
FROM python:3.8-slim AS builder
WORKDIR /end2end
RUN python -m venv .venv && .venv/bin/pip install --no-cache-dir -U pip setuptools
COPY requirement.txt .
RUN .venv/bin/pip install --no-cache-dir -r requirement.txt && find /end2end/.venv ( -type d -a -name test -o -name tests \) -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) -exec rm -rf '{}' \+


# Stage 2 - Copy only necessary files to the runner stage
FROM python:3.8-alpine
WORKDIR /end2end
COPY --from=builder /end2end /end2end
COPY infer.py .
ENV PATH="/end2end/.venv/bin:$PATH"
CMD ["python", "infer.py"]

# FROM python:3.8-slim
# COPY . /end2end
# WORKDIR /end2end
# # RUN pip install --upgrade pip
# RUN pip install --upgrade pip --no-cache-dir -r requirement.txt
# ENTRYPOINT ["python3",  "infer.py"]
