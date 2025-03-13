from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import meetings, users, topics, transcriptions

app = FastAPI(
    title="Government Meeting Tracker",
    description="An application to track government meetings and notify users when topics of interest are discussed.",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(meetings.router, prefix="/api", tags=["meetings"])
app.include_router(users.router, prefix="/api", tags=["users"])
app.include_router(topics.router, prefix="/api", tags=["topics"])
app.include_router(transcriptions.router, prefix="/api", tags=["transcriptions"])

@app.get("/", tags=["root"])
async def root():
    """Health check endpoint"""
    return {"status": "online"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 