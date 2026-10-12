# DocumentAgent Mobile App

React Native (Expo SDK 51) teacher grading client for the DocumentAgent backend.

## Quick start

```bash
cd mobile
npm install
cp .env.example .env
# Edit .env: set EXPO_PUBLIC_API_BASE_URL to your FastAPI server
npx expo start
```

Scan the QR code with Expo Go (Android/iOS) or press `a` for Android emulator.

## Screens

| Screen | Route | Purpose |
|---|---|---|
| Home | `/` | Quick-start, backend status, ledger count |
| Capture | `/capture` | Camera / gallery / PDF upload; student label |
| Grading Result | `/grading-result` | Extracted text, marks entry, share |
| Mark Sheet Review | `/mark-sheet-review` | Form template selector, field edit, PDF export |
| Grade Ledger | `/grade-ledger` | Session history, class average, CSV export |
| Settings | `/settings` | Backend toggle, API key, privacy notice |

## Building for Android

```bash
npx eas build --platform android --profile preview
```

Requires `EXPO_TOKEN` environment variable (set in EAS project settings).

## Environment variables

| Variable | Description |
|---|---|
| `EXPO_PUBLIC_API_BASE_URL` | FastAPI backend base URL (default: `http://localhost:8000`) |
