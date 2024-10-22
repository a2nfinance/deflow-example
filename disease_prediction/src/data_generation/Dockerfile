FROM python:slim
 
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
# For testing only
# COPY generation.py /app/generation.py  
# CMD ["tail", "-f", "/dev/null"]