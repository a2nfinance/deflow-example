FROM python:slim
 
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# For testing only
# COPY . /app
# CMD ["tail", "-f", "/dev/null"]