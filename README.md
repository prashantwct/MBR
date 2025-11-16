
Movebank full-stack deploy (Frontend + FastAPI backend) for Render.com
====================================================================

Structure:
- frontend/   -> Vite + React app (calls /api/download-study on the same origin)
- backend/    -> FastAPI proxy that downloads Movebank CSV and streams to client
- render.yaml -> Render deployment configuration (edit repo URL and backend URL)

Local development:
1. Frontend:
   cd frontend
   npm install
   npm run dev

2. Backend:
   cd backend
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8000

Deploy to Render:
- Create a Git repo with this content and push to GitHub.
- Update render.yaml: replace <your-repo-url> and backend URL placeholders.
- Go to Render.com -> New -> Deploy from repo -> select render.yaml
- In Render dashboard, set environment variables MOVEBANK_USER and MOVEBANK_PASS (do NOT commit them).

Notes:
- For production, restrict CORS origins instead of "*".
- Ensure backend URL is set in frontend (VITE_API_BASE) or use relative paths if deploying backend & frontend under same domain.
