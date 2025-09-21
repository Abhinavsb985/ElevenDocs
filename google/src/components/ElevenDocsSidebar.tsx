import { MessageSquare, Plus, Settings, LogOut, User, FileText, History } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ElevenDocsSidebarProps {
  onLogout: () => void;
  onNewChat: () => void;
}

export const ElevenDocsSidebar = ({ onLogout, onNewChat }: ElevenDocsSidebarProps) => {
  const chatHistory = [
    { id: 1, title: "Rental Agreement Analysis", time: "2 hours ago", type: "rental" },
    { id: 2, title: "Employment Contract Review", time: "Yesterday", type: "contract" },
    { id: 3, title: "Terms of Service Summary", time: "2 days ago", type: "terms" },
    { id: 4, title: "Privacy Policy Highlights", time: "1 week ago", type: "privacy" },
  ];

  return (
    <div className="w-full h-full flex flex-col bg-sidebar-bg">
      {/* Header - Fixed */}
      <div className="p-3 flex-shrink-0">
        <Button 
          onClick={onNewChat}
          className="w-full bg-transparent hover:bg-accent border border-border text-foreground flex items-center gap-3 transition-all duration-200 h-10 rounded-lg"
        >
          <Plus size={18} />
          <span className="font-medium">New chat</span>
        </Button>
      </div>

      {/* Chat History - Scrollable */}
      <div className="flex-1 overflow-y-auto px-2 pb-2 min-h-0 sidebar-scroll">
        <div className="space-y-1">
          {chatHistory.map((chat) => (
            <button
              key={chat.id}
              className="w-full text-left p-3 rounded-lg hover:bg-accent transition-colors group text-left"
            >
              <div className="flex items-start gap-3">
                <MessageSquare size={16} className="mt-0.5 text-muted-foreground flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-foreground truncate leading-5">
                    {chat.title}
                  </p>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* User Menu - Fixed */}
      <div className="p-3 flex-shrink-0 border-t border-border">
        <div className="space-y-1">
          <button className="w-full flex items-center gap-3 p-2 rounded-lg hover:bg-accent transition-colors text-left">
            <User size={16} className="text-muted-foreground" />
            <span className="text-sm text-foreground">Profile</span>
          </button>
          
          <button className="w-full flex items-center gap-3 p-2 rounded-lg hover:bg-accent transition-colors text-left">
            <Settings size={16} className="text-muted-foreground" />
            <span className="text-sm text-foreground">Settings</span>
          </button>
          
          <button 
            onClick={onLogout}
            className="w-full flex items-center gap-3 p-2 rounded-lg hover:bg-destructive/10 transition-colors text-left text-destructive"
          >
            <LogOut size={16} />
            <span className="text-sm">Sign Out</span>
          </button>
        </div>
      </div>
    </div>
  );
};