from fastapi import FastAPI

app = FastAPI(title="RAG App")


@app.get("/health")
def health():
    return {"status": "ok"}
