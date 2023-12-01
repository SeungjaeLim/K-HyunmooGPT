# python이라는 docker image를 받을건데, 그 버전은 3.11 버전으로 받겠다.
FROM python:3.11

WORKDIR /usr/src

RUN chmod -R u+x /usr/src

COPY ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade -r /requirements.txt

WORKDIR /usr/src/app

# Expose the port that Uvicorn will run on
EXPOSE 8000:8000

# Command to run Uvicorn
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]