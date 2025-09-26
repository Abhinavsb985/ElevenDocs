import { useState } from "react";
import { 
  Menu, 
  Plus, 
  MessageSquare, 
  Settings, 
  ChevronRight,
  Globe,
  Brain,
  Code,
  FileText,
  Calculator,
  Lightbulb
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface VerionSidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

const recentChats = [
  { id: 1, title: "Prototype Idea: City-Wid...", icon: Globe },
  { id: 2, title: "AI Traffic Management Syst...", icon: Brain },
  { id: 3, title: "Radar Traffic Signal Implem...", icon: Code },
  { id: 4, title: "Initial Greeting", icon: MessageSquare },
  { id: 5, title: "Calling Verion CLI in Terminal", icon: FileText },
  { id: 6, title: "Greeting and Offer of Help", icon: Lightbulb },
  { id: 7, title: "Generative AI for Legal Doc...", icon: FileText },
  { id: 8, title: "Python Summarization Agen...", icon: Code },
  { id: 9, title: "Declare: Verion-Inspired Le...", icon: Brain },
  { id: 10, title: "Build Legal AI Chatbot with ...", icon: Calculator },
];

export const VerionSidebar = ({ collapsed, onToggle }: VerionSidebarProps) => {
  const [hoveredItem, setHoveredItem] = useState<number | null>(null);

  return (
    <div 
      className={cn(
        "bg-sidebar-bg border-r border-border flex flex-col transition-all duration-300 ease-in-out relative",
        collapsed ? "w-16" : "w-80"
      )}
    >
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggle}
            className="h-8 w-8 text-muted-foreground hover:text-foreground"
          >
            <Menu className="h-4 w-4" />
          </Button>
          {!collapsed && (
            <div className="slide-in-left">
              <h1 className="text-xl font-semibold text-gradient">Verion</h1>
              <p className="text-xs text-muted-foreground">2.5 Pro</p>
            </div>
          )}
        </div>
      </div>

      {/* New Chat Button */}
      <div className="p-4">
        <Button
          variant="outline"
          className={cn(
            "w-full justify-start gap-3 bg-card hover:bg-suggestion-card-hover border-border",
            collapsed && "justify-center"
          )}
        >
          <Plus className="h-4 w-4" />
          {!collapsed && <span>New chat</span>}
        </Button>
      </div>

      {/* Recent Chats */}
      <div className="flex-1 overflow-y-auto px-2">
        {!collapsed && (
          <div className="px-2 mb-3">
            <h3 className="text-sm font-medium text-muted-foreground">Recent</h3>
          </div>
        )}
        
        <div className="space-y-1">
          {recentChats.map((chat, index) => {
            const Icon = chat.icon;
            return (
              <div
                key={chat.id}
                className={cn(
                  "group flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-all duration-200 slide-in-left hover:bg-suggestion-card-hover",
                  collapsed && "justify-center"
                )}
                style={{ animationDelay: `${index * 50}ms` }}
                onMouseEnter={() => setHoveredItem(chat.id)}
                onMouseLeave={() => setHoveredItem(null)}
              >
                <Icon className="h-4 w-4 text-muted-foreground group-hover:text-primary flex-shrink-0" />
                {!collapsed && (
                  <span className="text-sm text-muted-foreground group-hover:text-foreground truncate">
                    {chat.title}
                  </span>
                )}
                {hoveredItem === chat.id && !collapsed && (
                  <ChevronRight className="h-3 w-3 text-muted-foreground ml-auto opacity-0 group-hover:opacity-100 transition-opacity" />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Settings */}
      <div className="p-4 border-t border-border">
        <Button
          variant="ghost"
          className={cn(
            "w-full justify-start gap-3 text-muted-foreground hover:text-foreground",
            collapsed && "justify-center"
          )}
        >
          <Settings className="h-4 w-4" />
          {!collapsed && <span>Settings & help</span>}
        </Button>
      </div>
    </div>
  );
};