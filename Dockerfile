FROM python:3.11

WORKDIR /customer_churn

EXPOSE 8501

COPY . /customer_churn

RUN pip install -r requirements.txt

CMD streamlit run server_1.py