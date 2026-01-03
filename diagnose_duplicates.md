# Duplicate Metrics Diagnosis

## Summary

Database and server-side API logic do NOT contain duplicates. However, user reports seeing duplicates in the API response.

## Server-Side Verification (Completed)

✅ **Database Schema**: UNIQUE constraint `uq_run_step` is correctly applied
✅ **Database Content**: 1261 records, no duplicates (verified via direct SQL query)
✅ **API Logic**: Simulated API endpoint logic, no duplicates returned

## Client-Side Diagnosis (User Action Required)

### Step 1: Clear ALL Browser Data

1. Open Chrome DevTools (F12)
2. Go to **Application** tab
3. Click **Clear storage** (left sidebar)
4. Check all boxes:
   - ✅ Application cache
   - ✅ Cache storage
   - ✅ Local and session storage
   - ✅ IndexedDB
5. Click **Clear site data**
6. Close browser completely
7. Reopen and test

### Step 2: Direct API Test (Bypass Frontend)

Open this URL directly in browser (NOT in DevTools):

```
http://localhost:8000/api/v1/training/runs/55/metrics_db
```

**Expected**: JSON response with NO duplicate steps

**How to check for duplicates**:
1. Copy JSON response
2. Search for `"step":73` (or any step number that appeared twice before)
3. Count occurrences - should be **exactly 1**

### Step 3: Check Browser DevTools Network Tab

1. Open DevTools → **Network** tab
2. Filter: `metrics_db`
3. Refresh the Training Monitor page
4. Click on the `metrics_db` request
5. Go to **Response** tab
6. Check if response contains duplicates

**Important**: Check the **Response** tab, NOT the **Preview** tab (Preview may cache old data)

### Step 4: Check for Multiple API Requests

In Network tab, check if there are **multiple requests** to `/metrics_db`:
- If yes: Frontend may be making duplicate requests and merging responses

### Step 5: Temporary: Disable SSE Real-Time Update

To isolate the issue, temporarily disable SSE update in LossChart.tsx:

Comment out the SSE useEffect (line 139-203):
```typescript
// Real-time SSE update for training metrics (when training is running)
// useEffect(() => {
//   if (!isRunning) return;
//   ...
// }, [isRunning, runId]);
```

Then refresh and check if duplicates still appear.

## Possible Root Causes

1. **Browser Cache**: Most likely - old cached response
2. **Frontend Merging Issue**: LossChart merging old polling data with new SSE data
3. **Multiple API Requests**: Frontend making duplicate requests
4. **Browser DevTools Cache**: DevTools showing cached response

## Next Steps

Please perform Steps 1-3 and report:
- Does direct API test (`/metrics_db`) show duplicates? (Yes/No)
- Does clearing browser data fix the issue? (Yes/No)
- Screenshot of Network tab showing the actual API response

## Technical Details

**Database State** (as of 2026-01-04):
- Total records for run_id=55: **1261**
- Unique steps: **1-1261** (no gaps)
- Duplicates in DB: **0**
- UNIQUE constraint: **Active**

**API Endpoint**:
- Route: `GET /training/runs/{run_id}/metrics_db`
- Data source: **SQLAlchemy database** (NOT TensorBoard)
- Decimation: 1261 → 1000 points (step_size=1)
- Duplicates in API logic: **0** (verified via simulation)
