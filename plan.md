# Plan: PneumoScan AI - Premium Frontend Implementation

## Objective
Implement a professional, clinical, and premium AI-powered medical imaging dashboard based on the provided design specification.

## Deliverables
- [ ] **Refactored Backend**: Transition `app.py` to a FastAPI server to support a custom frontend.
- [ ] **Design System**: Comprehensive `index.css` with a medical color palette, modern typography (Inter), and smooth interactions.
- [ ] **Multi-page Layout**: Fixed sidebar navigation with Dashboard, History, Model Info, How It Works, and Disclaimer.
- [ ] **Core Features**: 
    - Drag-and-drop X-ray upload.
    - AI Analysis workflow with real-time feedback.
    - Prediction results with confidence scores.
    - Grad-CAM heatmap visualization.
- [ ] **Responsive Design**: Support for Desktop, Tablet, and Mobile.

## Acceptance Criteria
- The UI matches the "Clinical but modern" aesthetic described in the spec.
- The color system (Deep Indigo, Electric Violet, Medical Green/Red) is correctly implemented.
- The sidebar navigation works and switches between pages.
- The analysis workflow correctly triggers the backend model and displays results (Prediction + Grad-CAM).
- No placeholder functionality is implied; static pages are informational as requested.

## Edge Cases & Constraints
- Large image uploads (handled by client-side resizing or backend limits).
- Non-X-ray images (basic validation).
- Backend processing time (loader/spinner required).
- No actual persistent history (mock data used as per spec).

## Implementation Steps
- [ ] **Step 1: Backend Refactor**
    - Convert `app.py` to FastAPI.
    - Create `/predict` POST endpoint.
    - Serve static files (frontend).
- [ ] **Step 2: Frontend Foundation**
    - Create `frontend/` directory.
    - Initialize `index.html` (Main Shell).
    - Initialize `style.css` (Design System).
- [ ] **Step 3: Component Building**
    - Implement the Sidebar (Persistent).
    - Implement Card components.
    - Implement the Upload Zone.
- [ ] **Step 4: Page Content**
    - Dashboard logic (Upload -> Results).
    - Informational pages (Model Info, How it Works, etc.).
- [ ] **Step 5: Visual Polish**
    - Add gradients, glows, and micro-animations.
    - Ensure responsiveness.
- [ ] **Step 6: Integration & Verification**
    - Connect frontend to `/predict` endpoint.
    - Manual QA checks.

## Files & APIs Touched
- `app.py` (Backend logic & API)
- `frontend/index.html`
- `frontend/style.css`
- `frontend/main.js`

## Manual QA / Verification Steps
- [ ] Verify image upload works via drag-and-drop.
- [ ] Confirm the "Analyze" button triggers a request and shows a loading state.
- [ ] Check that results (Normal/Pneumonia) change the card color (Green/Red).
- [ ] Verify Grad-CAM image is displayed alongside the original.
- [ ] Test navigation between all 5 pages.
- [ ] Test mobile responsiveness (Chrome DevTools).

## Notes / Decisions
- **FastAPI** chosen over Gradio for maximum UI control.
- **Vanilla CSS** with CSS Variables for the design system.
- **Base64** encoding for images to avoid managing temporary files in the initial version.
