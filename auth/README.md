# Firebase Auth + Firestore Setup

Follow these steps to configure Firebase Authentication and Firestore for this app.

1) Create a Firebase project and Web App in Firebase Console. Enable Google and Email/Password providers.

2) In `google/` (next to `package.json`), create `.env.local` with:

VITE_FIREBASE_API_KEY=your_api_key
VITE_FIREBASE_AUTH_DOMAIN=your_auth_domain
VITE_FIREBASE_PROJECT_ID=your_project_id
VITE_FIREBASE_STORAGE_BUCKET=your_storage_bucket
VITE_FIREBASE_MESSAGING_SENDER_ID=your_messaging_sender_id
VITE_FIREBASE_APP_ID=your_app_id

3) Firestore rules (Console → Firestore → Rules):

rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
  }
}

4) Run locally:
- npm install
- npm run dev

On sign-in, a user document is created/updated at `users/{uid}` with: uid, email, displayName, provider.
