FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

WORKDIR /app

COPY ./app /app

RUN pip3 install -r requirements.txt --no-cache-dir

CMD ["uvicorn","backened:app","--host","0.0.0.0","--port","8000"]

