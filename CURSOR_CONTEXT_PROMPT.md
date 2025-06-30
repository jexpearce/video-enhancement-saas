# Video Enhancement SaaS - Context for Cursor AI Debugging Session

## 🎯 **CRITICAL: You'll be helping the user TEST and DEBUG this application**
The user will be **actively testing features and encountering bugs**. Your primary job is to:
- **Debug issues as they arise** 
- **Fix connection problems, API errors, and UI bugs**
- **Help get the full application working end-to-end**
- **Be patient and methodical with troubleshooting**

## 📋 **Project Overview**

### **Core Concept**
This is an **AI-powered video enhancement SaaS** that automatically:
1. **Analyzes talking-head videos** (like TikToks, educational content, news commentary)
2. **Detects emphasized words** in the speaker's audio using NLP
3. **Generates relevant visual context** (images, graphics, text overlays) 
4. **Composites everything** into an enhanced final video

**Target Use Case**: Content creators making videos about news, politics, education, or any topic who want automatic visual enhancement.

### **Current Architecture**

## 🔧 **Backend (FastAPI + Python)**
**Location**: `/backend/`
**Main Entry**: `backend/app/main.py`
**Status**: ✅ **RUNNING** on `localhost:8000`

### **Key Backend Components**:

1. **API Endpoints** (`backend/app/api/v1/`):
   - `videos.py` - Video upload/download endpoints
   - `jobs.py` - Job status/management 
   - `health.py` - Health checks

2. **Database** (`backend/app/database/`):
   - SQLAlchemy models for jobs, videos, images
   - Processing job tracking

3. **Services** (`backend/app/services/`):
   - Video processing pipeline
   - NLP analysis for emphasis detection
   - Image generation/curation
   - Video composition

### **Upload Flow**:
```
POST /api/v1/videos/upload
- Accepts: multipart/form-data with video file
- Query params: target_platform, user_id  
- Returns: job_id for tracking
- Starts background processing
```

### **Recently Fixed Issue**:
❌ **WAS**: Frontend sending `target_platform`/`user_id` as FormData
✅ **NOW**: Correctly sends them as query parameters

## 🎨 **Frontend (React + TypeScript + Vite)**
**Location**: `/frontend/`
**Status**: ✅ **RUNNING** on `localhost:5173` (or 5174 if 5173 busy)

### **Key Frontend Components**:

1. **Pages**:
   - `HomePage.tsx` - Main landing page with upload
   - `HistoryPage.tsx` - Job history and results

2. **Upload Components** (`frontend/src/components/upload/`):
   - `VideoUpload.tsx` - Main upload interface
   - `PreferencesPanel.tsx` - Platform/quality settings
   - `UploadZone.tsx` - Drag/drop file area

3. **API Service** (`frontend/src/services/api.ts`):
   - Axios-based API client
   - Handles uploads, job polling, downloads

### **Current State**:
- ✅ Frontend loads properly
- ✅ Upload form renders
- ✅ Backend connection established  
- ⚠️ **LIKELY BUGS REMAINING**: File validation, progress tracking, job polling, results display

## 🐛 **Expected Debugging Areas**

### **High Priority Issues to Watch For**:

1. **File Upload Issues**:
   - File type validation errors
   - Upload progress not working
   - Large file timeouts

2. **API Communication**:
   - CORS issues
   - Request format problems  
   - Error handling gaps

3. **Job Processing**:
   - Background job failures
   - Status polling not working
   - Results not displaying

4. **Frontend State Management**:
   - Upload state inconsistencies
   - Error message display
   - Loading states

### **Common Error Patterns**:
- `422 Unprocessable Entity` - Usually parameter format issues
- CORS errors - Backend middleware config
- File type rejections - MIME type validation
- Database connection issues - Missing dependencies

## 🛠 **Development Setup**

### **Starting the Application**:

**Backend** (Terminal 1):
```bash
cd backend
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend** (Terminal 2):  
```bash
cd frontend
npm run dev
```

### **Key URLs**:
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/api/v1/health`

## 📁 **Project Structure**
```
video-enhancement-saas/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app entry
│   │   ├── api/v1/              # API endpoints
│   │   ├── database/            # DB models/connection  
│   │   ├── services/            # Processing logic
│   │   └── models/              # Pydantic schemas
│   ├── requirements.txt
│   └── Dockerfile
└── frontend/
    ├── src/
    │   ├── components/          # React components
    │   ├── services/            # API client
    │   ├── types/               # TypeScript types
    │   └── pages/               # Page components
    ├── package.json
    └── vite.config.ts
```

## 🎯 **Testing Strategy**

### **Step-by-Step Testing Approach**:

1. **Basic Connectivity**:
   - ✅ Backend health check responds
   - ✅ Frontend loads without errors
   - ✅ CORS allows requests

2. **File Upload Flow**:
   - Upload small test video (.mp4)
   - Verify job creation
   - Check parameter passing
   - Monitor console for errors

3. **Processing Pipeline**:
   - Track job status changes  
   - Verify background processing starts
   - Check for processing errors
   - Validate output generation

4. **Results & Download**:
   - Job completes successfully
   - Results display properly
   - Download links work
   - File outputs are valid

## ⚡ **Quick Debug Commands**

```bash
# Check if backend is running
curl http://localhost:8000/api/v1/health

# Test upload endpoint
curl -X POST "http://localhost:8000/api/v1/videos/upload?target_platform=tiktok&user_id=test" \
  -F "file=@test_video.mp4"

# Check frontend console
# Browser Dev Tools > Console (look for React/API errors)

# Backend logs  
# Check terminal running uvicorn for error messages
```

## 🔥 **IMPORTANT REMINDERS FOR DEBUGGING**

1. **The user WILL encounter bugs** - this is expected and normal
2. **Be systematic** - fix one issue at a time  
3. **Check both frontend console AND backend logs** for errors
4. **Test end-to-end** - upload → processing → results
5. **File format issues are common** - validate MIME types carefully
6. **CORS issues may resurface** - check middleware config
7. **Database errors** - ensure models/migrations are correct

## 💡 **Success Criteria**
- User can upload a video file
- Job is created and tracked properly  
- Processing completes (even if simplified)
- Results are displayed/downloadable
- No critical console errors

---

**Good luck debugging! The foundation is solid, now it's time to iron out the details.** 