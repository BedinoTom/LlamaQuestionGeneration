FROM python:3.10

WORKDIR /

COPY ./requirements.txt /requirements.txt

RUN pip install -r requirements.txt

COPY ./main.py /main.py
COPY ./models /models

CMD ["flask", "--app", "main", "run"]