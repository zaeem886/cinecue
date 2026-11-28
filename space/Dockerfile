# CineCue website on a Hugging Face Docker Space. scripts/deploy_space.py uploads this file
# with app.py, src/, the results summaries and posters.csv. The fitted models are downloaded
# at startup from the latest release of the private GitHub repository named by
# CINECUE_MODEL_REPO, using the Space's GITHUB_TOKEN secret.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements-app.txt .
RUN pip install --no-cache-dir --user -r requirements-app.txt

COPY --chown=user . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
