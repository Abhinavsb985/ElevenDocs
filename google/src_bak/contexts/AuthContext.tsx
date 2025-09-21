import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { initializeApp, type FirebaseApp } from 'firebase/app';
import { getAuth, GoogleAuthProvider, signInWithPopup, onAuthStateChanged, createUserWithEmailAndPassword, signInWithEmailAndPassword, signOut, type User as FirebaseUser } from 'firebase/auth';
import { getFirestore, doc, getDoc, setDoc, serverTimestamp } from 'firebase/firestore';

interface AppUser {
  uid: string;
  email: string | null;
  displayName: string | null;
  photoURL?: string | null;
  provider: string | null;
}

interface AuthContextType {
  user: AppUser | null;
  isAuthenticated: boolean;
  idToken: string | null;
  loading: boolean;
  showLoginModal: boolean;
  login: () => Promise<void>;
  logout: () => Promise<void>;
  signup: (username: string, email: string, password: string) => Promise<void>;
  signin: (email: string, password: string) => Promise<void>;
  setShowLoginModal: (show: boolean) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};

let app: FirebaseApp | undefined;
let auth = undefined as unknown as ReturnType<typeof getAuth>;
let db = undefined as unknown as ReturnType<typeof getFirestore>;

const ensureFirebase = () => {
  if (!app) {
    app = initializeApp(firebaseConfig);
    auth = getAuth(app);
    db = getFirestore(app);
  }
};

const mapFirebaseUser = (fbUser: FirebaseUser | null): AppUser | null => {
  if (!fbUser) return null;
  const provider = fbUser.providerData[0]?.providerId || null;
  return {
    uid: fbUser.uid,
    email: fbUser.email,
    displayName: fbUser.displayName,
    photoURL: fbUser.photoURL,
    provider,
  };
};

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<AppUser | null>(null);
  const [idToken, setIdToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [showLoginModal, setShowLoginModal] = useState(false);

  useEffect(() => {
    ensureFirebase();
    const unsubscribe = onAuthStateChanged(auth, async (fbUser) => {
      if (fbUser) {
        const token = await fbUser.getIdToken();
        setIdToken(token);
        const appUser = mapFirebaseUser(fbUser);
        setUser(appUser);
        setShowLoginModal(false);

        // Ensure user profile exists in Firestore
        try {
          const userRef = doc(db, 'users', fbUser.uid);
          const snapshot = await getDoc(userRef);
          if (!snapshot.exists()) {
            await setDoc(userRef, {
              uid: fbUser.uid,
              email: fbUser.email,
              displayName: fbUser.displayName,
              provider: fbUser.providerData[0]?.providerId || null,
              createdAt: serverTimestamp(),
              updatedAt: serverTimestamp(),
            });
          } else {
            // Update provider/displayName if changed
            await setDoc(userRef, {
              uid: fbUser.uid,
              email: fbUser.email,
              displayName: fbUser.displayName,
              provider: fbUser.providerData[0]?.providerId || null,
              updatedAt: serverTimestamp(),
            }, { merge: true });
          }
        } catch (e) {
          console.warn('Failed to ensure user profile:', e);
        }
      } else {
        setUser(null);
        setIdToken(null);
        setShowLoginModal(true);
      }
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  const login = async () => {
    ensureFirebase();
    const provider = new GoogleAuthProvider();
    await signInWithPopup(auth, provider);
  };

  const signup = async (username: string, email: string, password: string) => {
    ensureFirebase();
    const cred = await createUserWithEmailAndPassword(auth, email, password);
    // Optionally set displayName client-side
    const fbUser = cred.user;
    const userRef = doc(db, 'users', fbUser.uid);
    await setDoc(userRef, {
      uid: fbUser.uid,
      email: fbUser.email,
      displayName: username || fbUser.displayName || null,
      provider: 'password',
      createdAt: serverTimestamp(),
      updatedAt: serverTimestamp(),
    }, { merge: true });
  };

  const signin = async (email: string, password: string) => {
    ensureFirebase();
    await signInWithEmailAndPassword(auth, email, password);
  };

  const logout = async () => {
    ensureFirebase();
    await signOut(auth);
  };

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user,
    idToken,
    loading,
    showLoginModal,
    login,
    logout,
    signup,
    signin,
    setShowLoginModal,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
