# Docker Setup Notes — PDF RAG Assistant

## 1. Why Docker was added

Docker packages the PDF RAG Assistant and its Python dependencies into a container.

Without Docker:

- Python must be installed locally.
- Dependencies must be installed manually.
- Python/library versions can differ between machines.

With Docker:

- The application runs inside a consistent environment.
- All Python dependencies are installed inside the image.
- The application can be started with the same Docker commands on another machine.

The project uses:

- FastAPI
- Uvicorn
- LangChain
- FAISS
- HuggingFace Embeddings
- Gemini LLM

---

## 2. Docker files

The Docker setup contains:

```text
Dockerfile
.dockerignore
```

The project also uses persistent local directories:

```text
data/pdfs/
vectorstore/faiss_index/
```

These contain application data and should not be lost when the container is recreated.

---

## 3. Dockerfile

The Dockerfile defines how the Docker image is built.

Example:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Important parts

### `FROM`

```dockerfile
FROM python:3.12-slim
```

Uses Python 3.12 as the base image.

`slim` is used because it is smaller than the full Python image.

---

### `WORKDIR`

```dockerfile
WORKDIR /app
```

Sets `/app` as the working directory inside the container.

The project is therefore available inside:

```text
/app
```

---

### Copy requirements

```dockerfile
COPY requirements.txt .
```

Copies `requirements.txt` into `/app`.

---

### Install dependencies

```dockerfile
RUN pip install --no-cache-dir -r requirements.txt
```

Installs all Python dependencies required by the project.

`--no-cache-dir` prevents pip from keeping its package cache, helping reduce image size.

---

### Copy project

```dockerfile
COPY . .
```

Copies the project files into the container.

---

### Expose port

```dockerfile
EXPOSE 8000
```

Documents that the application uses port `8000`.

It does not actually publish the port to the host by itself.

---

### Start FastAPI

```dockerfile
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Starts the FastAPI application.

`0.0.0.0` is important because the application must accept connections from outside the container.

---

# 4. .dockerignore

`.dockerignore` prevents unnecessary files from being copied into the Docker image.

Example:

```text
.venv/
__pycache__/
*.pyc
.git/
.gitignore
.env
.ipynb_checkpoints/
.DS_Store
vectorstore/
data/
```

Important:

`.env` should not be copied into the image because it contains secrets.

The local `data/` and `vectorstore/` directories are also excluded because they should be mounted/persisted separately rather than baked into the image.

---

# 5. Docker configuration in settings.py

The application uses `config/settings.py` for configuration.

Important settings include:

```python
upload_directory
max_upload_size_mb
vector_store_directory
google_api_key
environment
```

The application uses environment variables for sensitive configuration.

For example:

```text
GOOGLE_API_KEY
```

is loaded through:

```python
google_api_key: str = Field(...)
```

The `.env` file is used during local development.

It should NOT be committed to Git.

---

# 6. Build the Docker image

Build the image from the project root:

```bash
docker build -t pdf-rag-assistant .
```

### Meaning

```text
docker build
```

Builds a Docker image.

```text
-t pdf-rag-assistant
```

Assigns the image name.

```text
.
```

Uses the current directory as the Docker build context.

---

## 7. Check the image

```bash
docker images
```

Example:

```text
REPOSITORY         TAG       IMAGE ID
pdf-rag-assistant  latest    1b2e79f89cef
```

The image can be several GB because the project includes libraries such as:

- PyTorch
- Transformers
- Sentence Transformers
- FAISS
- SciPy
- LangChain

The image size is not the same thing as the size of one file on disk.

Docker stores images inside Docker's own storage system.

---

# 8. Run the container

Example:

```bash
docker run -d \
  --name pdf-rag-assistant \
  -p 8000:8000 \
  --env-file .env \
  -v "$(pwd)/data/pdfs:/app/data/pdfs" \
  -v "$(pwd)/vectorstore/faiss_index:/app/vectorstore/faiss_index" \
  pdf-rag-assistant
```

### Important options

### `-d`

Runs the container in detached/background mode.

---

### `--name`

```bash
--name pdf-rag-assistant
```

Gives the container a readable name.

---

### `-p`

```bash
-p 8000:8000
```

Maps:

```text
Host port 8000
        ↓
Container port 8000
```

Therefore the API can be accessed using:

```text
http://localhost:8000
```

---

### `--env-file`

```bash
--env-file .env
```

Loads environment variables from `.env`.

This allows the container to access:

```text
GOOGLE_API_KEY
```

without putting the API key into the Docker image.

---

### Volume for PDFs

```bash
-v "$(pwd)/data/pdfs:/app/data/pdfs"
```

Maps the local PDF directory to the container.

```text
Host
data/pdfs/
     ↓
Container
/app/data/pdfs/
```

This means uploaded PDFs survive container restarts/recreation.

---

### Volume for FAISS

```bash
-v "$(pwd)/vectorstore/faiss_index:/app/vectorstore/faiss_index"
```

Persists the FAISS files outside the container.

The directory contains:

```text
index.faiss
index.pkl
metadata.json
```

Therefore the FAISS index survives container recreation.

---

# 9. Check running containers

```bash
docker ps
```

Shows currently running containers.

Example:

```text
CONTAINER ID   IMAGE                      PORTS
00538f225745   pdf-rag-assistant:latest   0.0.0.0:8000->8000/tcp
```

---

# 10. Check all containers

```bash
docker ps -a
```

Unlike `docker ps`, this also shows stopped containers.

---

# 11. Check container logs

```bash
docker logs pdf-rag-assistant
```

Useful for checking:

- FastAPI startup
- Embedding model loading
- LLM initialization
- FAISS loading
- Exceptions
- Application shutdown

Follow logs continuously:

```bash
docker logs -f pdf-rag-assistant
```

Press:

```text
Ctrl + C
```

to stop following the logs.

---

# 12. Test the health endpoint

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "message": "PDF RAG Assistant API is running."
}
```

---

# 13. Test PDF upload

Example:

```bash
curl -X POST \
  -F "file=@data/pdfs/Bert_Model.pdf" \
  http://localhost:8000/upload
```

The API should return a successful upload response.

The uploaded PDF is stored in:

```text
data/pdfs/
```

because that directory is mounted as a Docker volume.

---

# 14. Test indexing

Example:

```bash
curl -X POST \
  http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"filename":"Bert_Model.pdf"}'
```

This creates:

```text
index.faiss
index.pkl
metadata.json
```

inside:

```text
vectorstore/faiss_index/
```

---

# 15. Test querying

Example:

```bash
curl -X POST \
  http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is BERT?"}'
```

The API should return:

- generated answer
- source page
- source chunk ID

---

# 16. Verify FAISS persistence

Check the local directory:

```bash
ls -lh vectorstore/faiss_index/
```

Expected:

```text
index.faiss
index.pkl
metadata.json
```

Because this directory is mounted into the container, the files are stored on the host machine.

---

# 17. Verify PDF persistence

Check:

```bash
ls -lh data/pdfs/
```

Uploaded PDFs should remain there.

This is important because files stored only inside a container can disappear when the container is removed.

---

# 18. Restart the container

Stop:

```bash
docker stop pdf-rag-assistant
```

Start:

```bash
docker start pdf-rag-assistant
```

Check:

```bash
docker ps
```

Then test:

```bash
curl http://localhost:8000/health
```

The API should become available again after the application finishes starting.

The first few seconds after startup may return a connection error while the embedding model and LLM are initializing.

---

# 19. Test persistence after restart

After restarting the container:

```bash
curl http://localhost:8000/health
```

Then test the query again:

```bash
curl -X POST \
  http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is BERT?"}'
```

If the query works without recreating the FAISS index, persistence is working correctly.

---

# 20. Remove the container

To remove the container:

```bash
docker stop pdf-rag-assistant
docker rm pdf-rag-assistant
```

This removes the container but does not remove the Docker image.

Because PDFs and FAISS are stored through host volumes, they remain on the host.

---

# 21. Remove the Docker image

Check images:

```bash
docker images
```

Remove the image:

```bash
docker rmi pdf-rag-assistant:latest
```

The container must be removed first.

---

# 22. Docker cleanup

Check unused Docker resources:

```bash
docker system df
```

This shows Docker disk usage.

To remove unused resources:

```bash
docker system prune
```

Be careful with aggressive cleanup commands because they can remove unused containers, networks, images, and build cache.

---

# 23. Important persistence concept

There are three different things to remember:

$$
Docker Image \\
\downarrow \\
Application + Python dependencies \\
$$

$$
Docker Container \\
\downarrow \\
Running instance of the image \\
$$

$$
Host Volumes \\
\downarrow \\
Persistent application data
$$

For this project:

$$
\text{data/pdfs/} \\
\downarrow \\
\text{PDF persistence}
$$

and

$$
\text{vectorstore/faiss\_index/} \\
\downarrow \\
\text{FAISS persistence}
$$

The image contains the application.

The container runs the application.

The host directories preserve the application's data.

---

# 24. Environment variables

The API key should be provided through `.env`.

Example:

```text
GOOGLE_API_KEY=your_api_key_here
```

Do not:

- commit `.env`
- copy `.env` into the Docker image
- hard-code the API key in Python
- hard-code the API key in the Dockerfile

`.gitignore` contains:

```text
.env
```

and `.dockerignore` contains:

```text
.env
```

---

# 25. Final Docker architecture

```text
                         Host Machine
                              |
                +-------------+-------------+
                |                           |
          data/pdfs/              vectorstore/faiss_index/
                |                           |
                | Docker Volumes            |
                |                           |
                +-------------+-------------+
                              |
                       Docker Container
                              |
                       PDF RAG Assistant
                              |
                     +--------+--------+
                     |                 |
                  FastAPI           RAG Pipeline
                     |                 |
                  Port 8000        FAISS + Gemini
```

The overall flow is:

$$
\text{Client} \\
\downarrow \\
\text{localhost:8000} \\
\downarrow \\
\text{Docker Container} \\
\downarrow \\
\text{FastAPI} \\
\downarrow \\
\text{PDF RAG Pipeline} \\
\downarrow \\
\text{FAISS + Embeddings + Gemini} \\
$$

---

# 26. Useful Docker commands

### Start container

```bash
docker start pdf-rag-assistant
```

### Stop container

```bash
docker stop pdf-rag-assistant
```

### Restart container

```bash
docker restart pdf-rag-assistant
```

### View running containers

```bash
docker ps
```

### View all containers

```bash
docker ps -a
```

### View logs

```bash
docker logs pdf-rag-assistant
```

### Follow logs

```bash
docker logs -f pdf-rag-assistant
```

### Build image

```bash
docker build -t pdf-rag-assistant .
```

### List images

```bash
docker images
```

### Remove container

```bash
docker rm pdf-rag-assistant
```

### Remove image

```bash
docker rmi pdf-rag-assistant:latest
```

### Check Docker disk usage

```bash
docker system df
```

---

# 27. Final Docker setup

The final Docker deployment provides:

- Containerized FastAPI application
- Python 3.12 environment
- Reproducible dependency installation
- Environment variable support
- PDF upload persistence
- FAISS persistence
- Container restart support
- Health endpoint
- API access through port 8000

The important files are:

```text
Dockerfile
.dockerignore
requirements.txt
config/settings.py
```

and the persistent directories are:

```text
data/pdfs/
vectorstore/faiss_index/
```

The Docker deployment was tested for:

1. Docker image build
2. Container startup
3. `/health` endpoint
4. PDF upload
5. FAISS indexing
6. `/query` endpoint
7. FAISS persistence
8. PDF persistence
9. Container restart
10. Environment variable loading
11. Final Docker cleanup