import LandingPage from "@/components/LandingPage";
import { AuthForm } from "@/components/AuthForm";
import { useAuth } from "@/contexts/AuthContext";
import SearchBar from "../components/SearchBar";

const Index = () => {
  const { user, isAuthenticated, loading } = useAuth();

  const handleLogin = () => {
    // Login handled by AuthForm
  };

  const handleLogout = () => {
    // This will be handled by the LandingPage component
  };

  const handleStartChat = (query: string, files: File[] = []) => {
    // This is now handled within the LandingPage component
    // No need to transition to a different page
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <AuthForm onLogin={handleLogin} />;
  }

  return (
    <LandingPage 
      onStartChat={handleStartChat}
      isAuthenticated={isAuthenticated}
      onLogout={handleLogout}
    />
  );
};

export default Index;
