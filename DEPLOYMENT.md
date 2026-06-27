# Deployment Guide

This deploys the project for **free**:

- **Backend (FastAPI)** → Hugging Face Spaces (Docker)
- **Frontend (Next.js)** → Vercel

Architecture in production:

```
Browser ──▶ Vercel (Next.js)  ──POST /chat (SSE)──▶  HF Space (FastAPI + FAISS + fastembed)
                                                          └──▶ HF Inference API (LLM)
```

Deploy in this order: **backend first** (so you have its URL for the frontend),
**frontend second**, then **point CORS back at the frontend**.

---

## 1. Backend → Hugging Face Space (Docker)

A Space is its own git repo. Because our backend lives in a subfolder of the
main repo, we create a Space and push the **contents of `backend/`** into it.

### 1a. Create the Space
1. Go to https://huggingface.co/new-space
2. Name it e.g. `center-desk-rag-api`.
3. **Space SDK:** choose **Docker** → **Blank**.
4. Visibility: Public (free). Create the Space.

### 1b. Push the backend code into the Space
From a terminal, in a folder **outside** this project (so the two git repos
don't collide):

```bash
# Clone the empty Space repo (use the URL shown on your Space page)
git clone https://huggingface.co/spaces/<your-username>/center-desk-rag-api
cd center-desk-rag-api

# Copy the backend contents in. Do NOT copy .env, .venv, vector_store, eval, or
# scripts. From this Space folder, with <PROJECT> = path to this project:
cp -r "<PROJECT>/backend/app" .
cp -r "<PROJECT>/backend/data" .
cp "<PROJECT>/backend/Dockerfile" .
cp "<PROJECT>/backend/.dockerignore" .
cp "<PROJECT>/backend/requirements.txt" .

# Use the Space metadata file as the Space's README.md (it has the YAML
# frontmatter HF needs: sdk: docker, app_port: 7860)
cp "<PROJECT>/backend/deploy/hf_space_README.md" README.md

git add -A
git commit -m "Deploy Center Desk RAG API"
git push
```

> When git asks for a password on push, use a Hugging Face **access token**
> (Settings → Access Tokens) with *write* permission, not your account password.

### 1c. Set the Space secrets
On the Space page → **Settings** → **Variables and secrets**:
- `HF_TOKEN` — **secret** — your HF Inference token (the LLM uses it).
- `CORS_ORIGINS` — **variable** — leave as a placeholder for now (e.g.
  `http://localhost:3000`); you'll set the real Vercel URL in step 3.

The Space will build the Docker image (installs deps, downloads the embedding
model, builds the FAISS index) and start. First build takes a few minutes.

### 1d. Verify
Your API base URL is `https://<your-username>-center-desk-rag-api.hf.space`.
Open `…/health` in a browser — it should return `{"status":"ok","indexed_entries":220,...}`.

---

## 2. Frontend → Vercel

1. Go to https://vercel.com → **Add New… → Project** → import this GitHub repo.
2. **Root Directory:** set to `frontend` (click *Edit* and pick the folder).
   Vercel auto-detects Next.js.
3. **Environment Variables:** add
   - `NEXT_PUBLIC_API_URL` = your Space URL from step 1d
     (e.g. `https://<your-username>-center-desk-rag-api.hf.space`)
4. **Deploy.** Vercel builds and gives you a URL like
   `https://center-desk-assistant-rag.vercel.app`.

> `NEXT_PUBLIC_*` variables are baked in at **build time**. If you change the URL
> later, trigger a redeploy (Deployments → ⋯ → Redeploy).

---

## 3. Connect CORS (backend ⇄ frontend)

Back on the **Space** → Settings → Variables and secrets, set:
- `CORS_ORIGINS` = your Vercel URL (e.g. `https://center-desk-assistant-rag.vercel.app`)

Save — the Space restarts. (Comma-separate if you want to keep localhost too,
e.g. `https://…vercel.app,http://localhost:3000`.)

Now open your Vercel URL and ask a question — it should stream a grounded answer.

---

## Notes & gotchas
- **Cold starts:** free HF Spaces sleep after inactivity; the first request after
  idle takes a few seconds to wake.
- **Why CORS matters here:** the backend calls the HF Inference API with *your*
  token, so restricting origins stops other websites from spending your quota
  through your API. Keep `CORS_ORIGINS` limited to your own frontend.
- **Local dev still works** unchanged: backend on `:8000`, frontend on `:3000`
  with `frontend/.env.local` pointing at `http://localhost:8000`.
- **Updating the deployment:** push new backend code to the Space repo; push
  frontend changes to GitHub and Vercel redeploys automatically.
