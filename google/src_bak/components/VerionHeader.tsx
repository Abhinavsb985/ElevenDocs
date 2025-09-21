import { Search, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import verionLogo from "@/assets/verion-logo.jpg";

interface VerionHeaderProps {
  onToggleSidebar: () => void;
  isProcessing?: boolean;
}

export const VerionHeader = ({ onToggleSidebar, isProcessing = false }: VerionHeaderProps) => {
  return (
    <header className="h-16 border-b border-border bg-background/80 backdrop-blur-sm flex items-center justify-between px-6">
      {/* Left section */}
      <div className="flex items-center gap-4">
        <Button
          variant="ghost"
          size="icon"
          className="text-muted-foreground hover:text-foreground"
        >
          <Search className="h-4 w-4" />
        </Button>
        
        <div className="flex items-center gap-3">
          {isProcessing ? (
            <div className="relative">
              <img
                src={verionLogo}
                alt="Verion"
                className="w-8 h-8 rounded-lg animate-bulge transition-all duration-300"
                style={{
                  filter: 'hue-rotate(0deg) saturate(1.2) brightness(1.1)',
                  animation: 'bulge 1.5s ease-in-out infinite, color-shift 2s ease-in-out infinite'
                }}
              />
              <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-orange-400 to-red-500 opacity-20 animate-pulse"></div>
            </div>
          ) : (
            <h1 className="text-xl font-semibold text-gradient">Verion</h1>
          )}
          <Button
            variant="ghost"
            className="text-sm text-muted-foreground hover:text-foreground gap-1 px-2"
          >
            2.5 Pro
            <ChevronDown className="h-3 w-3" />
          </Button>
        </div>
      </div>

      {/* Right section */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 text-sm">
          <span className="text-muted-foreground">PRO</span>
        </div>
        
        <Avatar className="h-8 w-8 ring-2 ring-primary/20">
          <AvatarImage src="" alt="User" />
          <AvatarFallback className="bg-gradient-primary text-primary-foreground font-semibold">
            A
          </AvatarFallback>
        </Avatar>
      </div>
    </header>
  );
};