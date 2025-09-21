import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardContent, CardFooter } from "@/components/ui/card";
import { GoogleLoginButton } from "./GoogleLoginButton";
import { useAuth } from "@/contexts/AuthContext";

interface AuthFormProps {
  onLogin: () => void;
}

export const AuthForm = ({ onLogin }: AuthFormProps) => {
  const [isSignUp, setIsSignUp] = useState(false);
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  
  const { signup, signin } = useAuth();

  const validateForm = () => {
    const newErrors: Record<string, string> = {};
    
    if (isSignUp && !username.trim()) {
      newErrors.username = "Username is required";
    } else if (isSignUp && username.length < 3) {
      newErrors.username = "Username must be at least 3 characters";
    }
    
    if (!email) {
      newErrors.email = "Email is required";
    } else if (!/\S+@\S+\.\S+/.test(email)) {
      newErrors.email = "Please enter a valid email";
    }
    
    if (!password) {
      newErrors.password = "Password is required";
    } else if (password.length < 8) {
      newErrors.password = "Password must be at least 8 characters";
    }
    
    if (isSignUp && password !== confirmPassword) {
      newErrors.confirmPassword = "Passwords don't match";
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateForm()) return;
    
    setLoading(true);
    try {
      if (isSignUp) {
        await signup(username, email, password);
      } else {
        await signin(email, password);
      }
      onLogin();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Authentication failed';
      setErrors({ general: errorMessage });
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSuccess = () => {
    onLogin();
  };

  const handleGoogleError = (error: string) => {
    setErrors({ general: error });
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Card className="w-full max-w-md auth-slide shadow-lg border-border">
        <CardHeader className="text-center space-y-2">
          <h1 className="text-2xl font-semibold text-foreground">
            {isSignUp ? "Create Account" : "Welcome back"}
          </h1>
          <p className="text-muted-foreground">
            {isSignUp 
              ? "Sign up to start chatting" 
              : "Sign in to continue your conversations"
            }
          </p>
        </CardHeader>
        
        <CardContent className="space-y-4">
          {/* Google Login Button */}
          <GoogleLoginButton 
            onSuccess={handleGoogleSuccess}
            onError={handleGoogleError}
          />
          
          {/* Divider */}
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <span className="w-full border-t" />
            </div>
            <div className="relative flex justify-center text-xs uppercase">
              <span className="bg-background px-2 text-muted-foreground">
                Or continue with
              </span>
            </div>
          </div>

          {/* General Error */}
          {errors.general && (
            <div className="text-sm text-destructive text-center bg-destructive/10 p-2 rounded">
              {errors.general}
            </div>
          )}

          {/* Traditional Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {isSignUp && (
              <div className="space-y-2">
                <Input
                  type="text"
                  placeholder="Username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className={`bg-input-bg border-input-border focus:border-input-border-focus transition-colors ${
                    errors.username ? "border-destructive" : ""
                  }`}
                />
                {errors.username && (
                  <p className="text-sm text-destructive">{errors.username}</p>
                )}
              </div>
            )}
            
            <div className="space-y-2">
              <Input
                type="email"
                placeholder="Email address"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className={`bg-input-bg border-input-border focus:border-input-border-focus transition-colors ${
                  errors.email ? "border-destructive" : ""
                }`}
              />
              {errors.email && (
                <p className="text-sm text-destructive">{errors.email}</p>
              )}
            </div>
            
            <div className="space-y-2">
              <Input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className={`bg-input-bg border-input-border focus:border-input-border-focus transition-colors ${
                  errors.password ? "border-destructive" : ""
                }`}
              />
              {errors.password && (
                <p className="text-sm text-destructive">{errors.password}</p>
              )}
            </div>
            
            {isSignUp && (
              <div className="space-y-2">
                <Input
                  type="password"
                  placeholder="Confirm password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className={`bg-input-bg border-input-border focus:border-input-border-focus transition-colors ${
                    errors.confirmPassword ? "border-destructive" : ""
                  }`}
                />
                {errors.confirmPassword && (
                  <p className="text-sm text-destructive">{errors.confirmPassword}</p>
                )}
              </div>
            )}
            
            <Button 
              type="submit" 
              className="w-full bg-primary hover:bg-primary-hover text-primary-foreground transition-colors"
              disabled={loading}
            >
              {loading ? "Please wait..." : (isSignUp ? "Create Account" : "Sign In")}
            </Button>
          </form>
        </CardContent>
        
        <CardFooter className="justify-center">
          <button
            type="button"
            onClick={() => {
              setIsSignUp(!isSignUp);
              setErrors({});
              setUsername("");
              setEmail("");
              setPassword("");
              setConfirmPassword("");
            }}
            className="text-primary hover:underline transition-all"
          >
            {isSignUp 
              ? "Already have an account? Sign in" 
              : "Don't have an account? Sign up"
            }
          </button>
        </CardFooter>
      </Card>
    </div>
  );
};