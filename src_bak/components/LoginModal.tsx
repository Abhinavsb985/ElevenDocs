import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { Loader2, X } from 'lucide-react';
import { Button } from '@/components/ui/button';

declare global {
  interface Window {
    google: any;
  }
}

interface LoginModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export const LoginModal: React.FC<LoginModalProps> = ({ isOpen, onClose, onSuccess }) => {
  const { loading, isAuthenticated } = useAuth();
  const [error, setError] = useState('');
  
  const googleSignInRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen && !isAuthenticated) {
      initializeGoogleSignIn();
    }
  }, [isOpen, isAuthenticated]);

  const initializeGoogleSignIn = async () => {
    try {
      if (window.google && googleSignInRef.current) {
        // Initialize Google Sign-In
        window.google.accounts.id.initialize({
          client_id: '690520166834-k3asho1m2e6kape6q75tnk1a15iaavl2.apps.googleusercontent.com',
          callback: handleGoogleResponse
        });

        // Render Sign In button
        window.google.accounts.id.renderButton(googleSignInRef.current, {
          theme: 'filled_blue',
          size: 'large',
          text: 'signin_with',
          shape: 'rectangular',
          width: 280
        });
      }
    } catch (error) {
      console.error('Error initializing Google Sign-In:', error);
    }
  };

  const handleGoogleResponse = async (response: any) => {
    try {
      // Decode the JWT token to get user info
      const payload = JSON.parse(atob(response.credential.split('.')[1]));
      
      const userData = {
        email: payload.email,
        name: payload.name,
        picture: payload.picture,
        sub: payload.sub
      };

      // Store in localStorage
      localStorage.setItem('google_id_token', response.credential);
      localStorage.setItem('google_user', JSON.stringify(userData));
      
      // Update auth context and close modal
      onSuccess();
      window.location.reload(); // Refresh to update auth state
    } catch (error) {
      console.error('Error processing Google response:', error);
      setError('Google authentication failed. Please try again.');
    }
  };

  if (!isOpen || isAuthenticated) {
    return null;
  }

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-2xl w-full max-w-md p-8 relative">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
        >
          <X size={20} />
        </button>

        <div className="text-center mb-8">
          <div className="h-16 w-16 bg-primary rounded-2xl flex items-center justify-center mx-auto mb-4">
            <span className="text-primary-foreground font-bold text-2xl">11D</span>
          </div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Welcome to ElevenDocs
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Sign in with Google to access AI-powered legal document analysis
          </p>
        </div>

        {loading && (
          <div className="text-center py-8">
            <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-primary" />
            <p className="text-gray-600 dark:text-gray-400">Loading...</p>
          </div>
        )}

        {!loading && (
          <div className="space-y-6">
            {/* Google Sign In Button */}
            <div className="flex justify-center">
              <div ref={googleSignInRef}></div>
            </div>

            {/* Error Display */}
            {error && (
              <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
              </div>
            )}

            {/* Footer */}
            <div className="text-center">
              <p className="text-xs text-gray-500 dark:text-gray-400">
                By continuing, you agree to our Terms of Service and Privacy Policy
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
