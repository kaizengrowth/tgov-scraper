# Government Meeting Tracker and Notification Application

## Overview
An application to track and manage government meetings, providing access to meeting details, agendas, and video recordings. The system includes topic-based subscriptions for users, automatic transcription and diarization of meeting videos, and SMS notifications when topics of interest are discussed.

## Core Features

### 1. Meeting Data Management
- Store meeting information including:
  - Meeting name
  - Date and time
  - Duration
  - Agenda links (PDF/document links)
  - Video recording links

### 2. Data Collection
- Automated scraping of meeting data from government websites
- Support for manual meeting data entry
- Validation of URLs for agenda and video links

### 3. API Endpoints 
#### Meeting Endpoints
- `GET /api/meetings` - List all meetings with pagination
- `GET /api/meetings/{id}` - Get specific meeting details
- `POST /api/meetings` - Create new meeting record
- `PUT /api/meetings/{id}` - Update meeting details
- `DELETE /api/meetings/{id}` - Delete meeting record

#### Filtering Endpoints
- `GET /api/meetings/search` - Search meetings by name/date
- `GET /api/meetings/recent` - Get recent meetings
- `GET /api/meetings/upcoming` - Get upcoming meetings

#### User and Subscription Endpoints
- `POST /api/users` - Register a new user
- `GET /api/users/{id}` - Get user information
- `PUT /api/users/{id}` - Update user information
- `POST /api/topics` - Create a topic subscription
- `GET /api/topics/user/{user_id}` - Get user's topics
- `DELETE /api/topics/{id}` - Delete a topic

#### Transcription Endpoints
- `GET /api/transcriptions/meeting/{meeting_id}` - Get meeting transcription
- `GET /api/transcriptions/search` - Search transcriptions by keyword/topic

### 4. Data Models
```python
class Meeting(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    meeting: str = Field(description="Name of the meeting")
    date: datetime = Field(description="Date and time of the meeting")
    duration: str = Field(description="Duration of the meeting")
    agenda: Optional[HttpUrl] = Field(None, description="URL to the meeting agenda")
    video: Optional[HttpUrl] = Field(None, description="URL to the meeting video")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        orm_mode = True

class User(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(description="User's full name")
    email: EmailStr = Field(description="User's email address")
    phone: str = Field(description="User's phone number for text notifications")
    topics: List[Topic] = Field(default_factory=list, description="User's topic subscriptions")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        orm_mode = True

class Topic(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(description="Topic keyword or phrase to monitor")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        orm_mode = True

class TranscriptSegment(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    meeting_id: UUID = Field(description="ID of the meeting")
    speaker: str = Field(description="Speaker identifier from diarization")
    start_time: float = Field(description="Start time in seconds from beginning of recording")
    end_time: float = Field(description="End time in seconds from beginning of recording")
    text: str = Field(description="Transcribed text segment")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        orm_mode = True

class Notification(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(description="ID of the user being notified")
    meeting_id: UUID = Field(description="ID of the related meeting")
    topic_id: UUID = Field(description="ID of the triggered subscription")
    segment_id: UUID = Field(description="ID of the transcript segment that matched")
    sent_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(description="Delivery status of the notification")

    class Config:
        orm_mode = True
```

### 5. Technical Requirements

#### Backend
- Python 3.9+
- FastAPI for REST API
- SQLAlchemy for database ORM
- Pydantic for data validation
- Alembic for database migrations
- PostgreSQL for data storage
- Async functionality for improved performance
- WhisperX for speech-to-text transcription and speaker diarization
- Twilio API for SMS notifications
- Background task processing with Celery or similar

#### Frontend
- React.js for user interface components
- Responsive design with Tailwind CSS or similar
- Form validation with Formik or similar
- Topic subscription management interface
- Topic suggestion functionality

#### Development Tools
- Poetry for dependency management
- Pre-commit hooks for code quality
- pytest for testing
- Black for code formatting
- isort for import sorting
- mypy for type checking
- flake8 for linting

### 6. Database Schema
```sql
CREATE TABLE meetings (
id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
meeting VARCHAR(255) NOT NULL,
date TIMESTAMP WITH TIME ZONE NOT NULL,
duration VARCHAR(50) NOT NULL,
agenda_url TEXT,
video_url TEXT,
created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_meetings_date ON meetings(date);
CREATE INDEX idx_meetings_name ON meetings(meeting);

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    phone VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);

CREATE TABLE topics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_topics_user ON topics(user_id);
CREATE INDEX idx_topics_topic ON topics(topic);

CREATE TABLE transcript_segments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meeting_id UUID NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    speaker VARCHAR(100) NOT NULL,
    start_time NUMERIC(10, 3) NOT NULL,
    end_time NUMERIC(10, 3) NOT NULL,
    text TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_transcript_meeting ON transcript_segments(meeting_id);
CREATE INDEX idx_transcript_text ON transcript_segments USING gin(to_tsvector('english', text));

CREATE TABLE notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    meeting_id UUID NOT NULL REFERENCES meetings(id),
    topic_id UUID NOT NULL REFERENCES topics(id),
    segment_id UUID NOT NULL REFERENCES transcript_segments(id),
    sent_at TIMESTAMP WITH TIME ZONE NOT NULL,
    status VARCHAR(50) NOT NULL
);

CREATE INDEX idx_notifications_user ON notifications(user_id);
CREATE INDEX idx_notifications_meeting ON notifications(meeting_id);
```

### 7. Project Structure
```
src/
├── api/
│   ├── __init__.py
│   ├── dependencies.py
│   └── routes/
│       ├── __init__.py
│       ├── meetings.py
│       ├── users.py
│       ├── topics.py
│       └── transcriptions.py
├── core/
│   ├── __init__.py
│   ├── config.py
│   └── security.py
├── db/
│   ├── __init__.py
│   ├── session.py
│   └── models/
│       ├── __init__.py
│       ├── meeting.py
│       ├── user.py
│       ├── topic.py
│       ├── transcription.py
│       └── notification.py
├── models/
│   ├── __init__.py
│   ├── meeting.py
│   ├── user.py
│   ├── topic.py
│   ├── transcription.py
│   └── notification.py
├── schemas/
│   ├── __init__.py
│   ├── meeting.py
│   ├── user.py
│   ├── topic.py
│   ├── transcription.py
│   └── notification.py
├── services/
│   ├── __init__.py
│   ├── meeting_service.py
│   ├── transcription_service.py
│   ├── notification_service.py
│   └── topic_service.py
├── scrapers/
│   ├── __init__.py
│   └── government_scraper.py
├── tasks/
│   ├── __init__.py
│   ├── transcription_tasks.py
│   └── notification_tasks.py
├── frontend/
│   ├── components/
│   │   ├── TopicForm.jsx
│   │   ├── MeetingList.jsx
│   │   └── TopicSelector.jsx
│   ├── pages/
│   │   ├── Home.jsx
│   │   ├── Topics.jsx
│   │   └── MeetingDetails.jsx
│   └── App.jsx
└── main.py
```

### 8. Testing Requirements
- Unit tests for all models and services
- Integration tests for API endpoints
- End-to-end tests for critical user flows
- Test coverage minimum: 80%
- Separate test database configuration
- Mocking external dependencies during tests
- CI/CD pipeline with automated test execution
- Regression test suite for bug fixes
- Performance testing for API endpoints
- Test fixtures for common test data
- Mock Twilio service for SMS testing
- Transcript analysis testing with predefined transcripts

### 9. Documentation
- OpenAPI/Swagger documentation for all endpoints
- README with setup instructions
- API documentation with example requests/responses
- Database schema documentation
- Development guidelines
- Contributing guidelines
- Environment setup guide
- Deployment procedures
- Architecture diagram
- Error codes and handling documentation
- Topic subscription workflow documentation
- Transcription process documentation

### 10. Deployment
- Docker containerization
- Docker Compose for local development
- Environment-based configuration
- Health check endpoints
- Logging configuration
- Monitoring setup
- CI/CD pipeline integration
- Database migration automation
- Backup and restore procedures
- Horizontal scaling capability
- Blue-green deployment strategy
- Environment variables management
- Secret management
- Asynchronous task queue configuration
- Transcription service integration management

### 11. Security Requirements
- Input validation
- URL validation for agenda and video links
- Rate limiting
- CORS configuration
- Environment variable management
- Secure connection (HTTPS) enforcement
- SQL injection prevention
- XSS protection
- CSRF protection
- Proper error handling without information leakage
- Security headers configuration
- Dependency vulnerability scanning
- Regular security audits
- Logging of security events
- Phone number validation and verification
- Data minimization for user information
- Privacy policy implementation

### 12. User Interface
- Clean, accessible web interface
- Responsive design for mobile and desktop
- Meeting search and filter functionality
- Video playback integration
- PDF viewer for agendas
- Calendar view of meetings
- Meeting notification system
- User preferences for regular meetings
- Topic management interface
- Notification history view
- Keyword suggestion feature
- User profile management
- Email and SMS preference settings

### 13. Performance Requirements
- API response time < 200ms for 95% of requests
- Support for at least 100 concurrent users
- Data scraping without impacting application performance
- Efficient database queries with proper indexing
- Caching strategy for frequently accessed data
- Pagination for large data sets
- Background processing for long-running tasks
- Database connection pooling
- Asynchronous processing of transcription jobs
- Batch processing for notifications
- Optimized full-text search for transcriptions

### 14. Subscription and Notification System
- Topic-based subscription management
- Real-time processing of new transcriptions
- Topic matching algorithm with support for:
  - Exact phrase matching
  - Stemming and lemmatization
  - Synonyms and related terms
  - Context-aware matching
- Notification throttling to prevent spam
- Scheduled delivery options (immediate vs. digest)
- User-configurable notification preferences
- Notification delivery confirmation and tracking
- Failed notification retry mechanism
- SMS delivery via Twilio API
- Optional email notification support

### 15. Transcription and Diarization System
- Automatic download of meeting videos
- Audio extraction from video files
- Speech-to-text processing using WhisperX:
  - Word-level timestamps for accurate topic identification
  - Speaker diarization to identify different speakers
  - VAD preprocessing to reduce hallucinations
- Timestamp generation for each speech segment
- Transcript correction and validation options
- Transcript search and indexing
- Topic detection in transcribed content
- Metadata extraction from transcripts
- Transcript export options (TXT, SRT, JSON)
- Speaker identification and naming (when available)
- Transcription quality metrics
