# UI Guide

## UI surfaces in this repo
1. Streamlit app:
   - [app.py](/c:/Users/Bhavesh/Documents/Python%20Scripts/Jeff/Cafe/milk-dashboard/app.py)
2. FastAPI static dashboard:
   - [index.html](/c:/Users/Bhavesh/Documents/Python%20Scripts/Jeff/Cafe/milk-dashboard/src/milk_dashboard/api/static/index.html)
   - [dashboard.js](/c:/Users/Bhavesh/Documents/Python%20Scripts/Jeff/Cafe/milk-dashboard/src/milk_dashboard/api/static/dashboard.js)
   - [dashboard.css](/c:/Users/Bhavesh/Documents/Python%20Scripts/Jeff/Cafe/milk-dashboard/src/milk_dashboard/api/static/dashboard.css)

There is no JSX/React build in the current stack. UI edits are Python (Streamlit) or plain HTML/CSS/JS.

## FastAPI dashboard change map
- Layout/content: `index.html`
- Interaction/state/fetch: `dashboard.js`
- Styling/tokens/theme: `dashboard.css`

Common edit patterns:
- Add new card or panel:
  1. Add markup in `index.html`.
  2. Bind element handles and render/update logic in `dashboard.js`.
  3. Add styles and responsive rules in `dashboard.css`.
- Add new API-driven metric:
  1. Add field extraction in `dashboard.js` API parse layer.
  2. Add render function update.
  3. Add tooltip/help text in `index.html` if needed.

## Streamlit app change map
- Main layout, controls, and charts all live in `app.py`.
- Use helper functions in the file to keep sections isolated.
- Keep any heavy data transforms outside UI callbacks when possible.

## API coupling points
- FastAPI app: [src/milk_dashboard/api/app.py](/c:/Users/Bhavesh/Documents/Python%20Scripts/Jeff/Cafe/milk-dashboard/src/milk_dashboard/api/app.py)
- Key endpoints for dashboard consumption include health and forecast routes.
- If response schema changes, update both:
  - backend response model/serializer,
  - `dashboard.js` parsing/render logic.

## UI change safety checklist
1. Verify desktop + mobile layout after CSS changes.
2. Confirm slider/input constraints match backend query limits.
3. Validate fallback behavior for missing/null API fields.
4. Test both UI surfaces when backend contract changes.
5. Keep style variables centralized in CSS root tokens when possible.
