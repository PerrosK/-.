READ ME

Κωνσταντίνος Νικόλαος Πέρρος ΑΜ:inf2021183
Ευάγγελος Τσόγκας ΑΜ:inf2021240

Οδηγίες για εκτέλεση μέσο Docker:
  Μέσα στο folder στον οποίο θα έχει το appara.py(ο κώδικας) θα φτιάξετε ενα file ονόματι Dockerfile που
  θα περιέχει τον παρακάτω κώδικα:
  
        FROM python:3.9-slim
        To μονοπάτι (root) για την εφαρμογή
        WORKDIR /appara.py
        Αντιγράφει το requirements.txt
        COPY requirements.txt .
        Καττεβάζει ό,τι υπάρχει στο requirements
        RUN pip install -r requirements.txt
        Copy the app code
        COPY . .
        Expose the port
        EXPOSE 8080
        Τρέξτε την εντολή για εκκίνηση της εφαρμογής
        CMD ["streamlit", "run", "appara.py"]

Μετά θα φτιάξεται ένα txt file ονόματι requirements.txt που θα περιέχει τα παρακάτω:

        streamlit
        pandas
        numpy
        matplotlib
        seaborn
        scikit-learn

Έπειτα, θα τρέξετε terminal(εμείς τα κάναμε ολα αυτά μέσο του vscode, όπου απο εκεί τρέξαμε το terminal)  και θα 
πληκτρολογήσετε τα παρακάτω με αυτή τη σειρά:

        1. docker build . -t container1/streamlit-app
        2. docker run -p 8080:8501 (και το id του image στο docker desktop στην περίπτωση μας ήταν:e506a5c431cce2b19c5633ef7b985b7ff1fe1148e9a3d5ed85ce443ad6de940f. Δεν ξέρουμε αν θα είναι το ίδιο)

Εμείς για να μπούμε στο web-page χρησιμοποιήσαμε την παρακάτω IP:

        http://localhost:8080/

