import { MessageSquare, Plus, Settings, LogOut, User } from "lucide-react";
import { Button } from "@/components/ui/button";

interface SidebarProps {
  onLogout: () => void;
  onNewChat: () => void;
}

export const Sidebar = ({ onLogout, onNewChat }: SidebarProps) => {
  const chatHistory = [
    { id: 1, title: "React vs Vue comparison", time: "2 hours ago" },
    { id: 2, title: "JavaScript best practices", time: "Yesterday" },
    { id: 3, title: "CSS Grid layout help", time: "2 days ago" },
    { id: 4, title: "Node.js deployment", time: "1 week ago" },
  ];

  return (
    <div className="w-64 bg-sidebar-bg border-r border-border flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <Button 
          onClick={onNewChat}
          className="w-full bg-primary hover:bg-primary-hover text-primary-foreground flex items-center gap-2 transition-colors"
        >
          <Plus size={16} />
          New Chat
        </Button>
      </div>

      {/* Chat History */}
      <div className="flex-1 overflow-y-auto p-2">
        <div className="space-y-1">
          {chatHistory.map((chat) => (
            <button
              key={chat.id}
              className="w-full text-left p-3 rounded-lg hover:bg-muted transition-colors group"
            >
              <div className="flex items-start gap-2">
                <MessageSquare size={16} className="mt-1 text-muted-foreground" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">
                    {chat.title}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {chat.time}
                  </p>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* User Menu */}
      <div className="border-t border-border p-4">
        <div className="space-y-2">
          <button className="w-full flex items-center gap-3 p-3 rounded-lg hover:bg-muted transition-colors text-left">
            <User size={16} className="text-muted-foreground" />
            <span className="text-sm text-foreground">Profile</span>
          </button>
          
          <button className="w-full flex items-center gap-3 p-3 rounded-lg hover:bg-muted transition-colors text-left">
            <Settings size={16} className="text-muted-foreground" />
            <span className="text-sm text-foreground">Settings</span>
          </button>
          
          <button 
            onClick={onLogout}
            className="w-full flex items-center gap-3 p-3 rounded-lg hover:bg-destructive/10 transition-colors text-left text-destructive"
          >
            <LogOut size={16} />
            <span className="text-sm">Sign Out</span>
          </button>
        </div>
      </div>
    </div>
  );
};