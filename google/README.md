# ElevenDocs Frontend

AI-powered legal document analyzer with Google authentication.

## Features

- 🔐 Google Sign-In authentication
- 📱 Responsive design with Tailwind CSS
- 🎨 Modern UI components with shadcn/ui
- 🔗 Backend API integration
- 📝 Document summarization

## Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Build for production:**
   ```bash
   npm run build
   ```

## Google Authentication

The app uses Google OAuth 2.0 for authentication. The Google Client ID is already configured in `index.html`.

### Authentication Flow

1. User clicks "Sign in with Google"
2. Google OAuth popup opens
3. User authorizes the app
4. ID token is received and stored
5. User is authenticated and can access protected features

### Backend Integration

- **Health Check:** `GET /health` (no auth required)
- **Summarize:** `POST /summarize` (requires Google ID token)

## Project Structure

```
src/
├── components/          # UI components
│   ├── ui/             # shadcn/ui components
│   ├── GoogleSignIn.tsx # Google authentication button
│   ├── UserMenu.tsx    # User menu dropdown
│   ├── LandingPage.tsx # Main landing page
│   └── BackendTest.tsx # Backend connection tester
├── contexts/           # React contexts
│   └── AuthContext.tsx # Authentication context
├── hooks/              # Custom hooks
│   └── useApi.ts       # API request hook
├── pages/              # Page components
│   ├── Index.tsx       # Main page
│   └── NotFound.tsx    # 404 page
└── lib/                # Utility functions
```

## Development

- **Frontend:** http://localhost:5173 (Vite default)
- **Backend:** http://localhost:5000 (Express server)

## Testing Backend Connection

Use the "Backend Connection Test" section at the bottom of the landing page to:

1. Test the health endpoint (no auth required)
2. Test the summarize endpoint (requires Google sign-in)

## Environment Variables

No environment variables needed for frontend. Google Client ID is hardcoded in `index.html`.

## Dependencies

- React 18
- TypeScript
- Tailwind CSS
- shadcn/ui components
- Google Sign-In API
- React Router
- React Query
