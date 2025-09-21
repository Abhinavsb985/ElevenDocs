import { User, LogOut, Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAuth } from "@/contexts/AuthContext";

interface UserMenuProps {
  isAuthenticated: boolean;
  onLogin: () => void;
  onLogout: () => void;
}

export const UserMenu = ({ isAuthenticated, onLogin, onLogout }: UserMenuProps) => {
  const { user } = useAuth();

  if (!isAuthenticated) {
    return null; // This should never happen with ProtectedRoute
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="h-10 w-10 rounded-full bg-secondary hover:bg-secondary-hover">
          <User size={18} className="text-secondary-foreground" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48 bg-card border-border">
        <DropdownMenuItem className="hover:bg-accent">
          <User size={16} className="mr-2" />
          {user?.email || 'Profile'}
        </DropdownMenuItem>
        <DropdownMenuItem className="hover:bg-accent">
          <Settings size={16} className="mr-2" />
          Settings
        </DropdownMenuItem>
        <DropdownMenuSeparator className="bg-border" />
        <DropdownMenuItem 
          onClick={onLogout}
          className="text-destructive hover:bg-destructive/10 cursor-pointer"
        >
          <LogOut size={16} className="mr-2" />
          Sign Out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};