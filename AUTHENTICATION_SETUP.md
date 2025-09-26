# Google Login Integration Setup

## Environment Variables

Create a `.env` file in the `google` directory with the following variables:

```
REACT_APP_GOOGLE_CLIENT_ID=your_google_client_id_here
REACT_APP_API_URL=http://127.0.0.1:8002
```

## Backend Environment Variables

Create a `.env` file in the `backend` directory with:

```
GOOGLE_CLIENT_ID=your_google_client_id_here
JWT_SECRET=your_jwt_secret_here
```

## Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google+ API
4. Go to "Credentials" and create OAuth 2.0 Client IDs
5. Add your domain to authorized origins
6. Copy the Client ID to your environment variables

## Installation

### Frontend
```bash
cd google
npm install
npm run dev
```

### Backend
```bash
cd backend
npm install
npm run dev
```

## Features

- ✅ Google OAuth login with "Continue with Google" button
- ✅ Traditional username/password signup
- ✅ Traditional email/password signin
- ✅ JWT token authentication
- ✅ Password hashing with bcrypt
- ✅ Clean, production-ready UI
- ✅ Error handling and validation
- ✅ Session management

## Usage

The authentication system supports both Google OAuth and traditional authentication methods. Users can:

1. Sign up with username, email, and password
2. Sign in with email and password
3. Continue with Google (OAuth)
4. Switch between signup and signin modes

All authentication methods return a JWT token that can be used for API requests.
