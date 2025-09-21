import { User, Bot, FileText } from "lucide-react";

interface MessageBubbleProps {
  message: string;
  isUser: boolean;
  timestamp: string;
  fileInfo?: {
    name: string;
    type: string;
    size: number;
  };
}

export const MessageBubble = ({ message, isUser, timestamp, fileInfo }: MessageBubbleProps) => {
  return (
    <div className={`flex gap-3 p-4 message-enter ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="flex-shrink-0 w-8 h-8 bg-ai-message border border-border rounded-full flex items-center justify-center">
          <Bot size={16} className="text-ai-message-foreground" />
        </div>
      )}
      
      <div className={`max-w-[70%] ${isUser ? 'order-first' : ''}`}>
        <div className={`p-3 rounded-2xl ${
          isUser 
            ? 'bg-user-message text-user-message-foreground ml-auto' 
            : 'bg-ai-message text-ai-message-foreground'
        }`}>
          {fileInfo && (
            <div className="flex items-center gap-2 mb-2 p-2 bg-muted/50 rounded-md">
              <FileText size={14} className="text-muted-foreground" />
              <span className="text-xs text-muted-foreground">
                {fileInfo.name} ({(fileInfo.size / 1024).toFixed(1)} KB)
              </span>
            </div>
          )}
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{message}</p>
        </div>
        <p className={`text-xs text-muted-foreground mt-1 ${isUser ? 'text-right' : 'text-left'}`}>
          {timestamp}
        </p>
      </div>
      
      {isUser && (
        <div className="flex-shrink-0 w-8 h-8 bg-user-message border border-border rounded-full flex items-center justify-center">
          <User size={16} className="text-user-message-foreground" />
        </div>
      )}
    </div>
  );
};

export const TypingIndicator = () => {
  return (
    <div className="flex gap-3 p-4 message-enter">
      <div className="flex-shrink-0 w-8 h-8 bg-ai-message border border-border rounded-full flex items-center justify-center">
        <Bot size={16} className="text-ai-message-foreground" />
      </div>
      
      <div className="bg-ai-message text-ai-message-foreground p-3 rounded-2xl">
        <div className="flex items-center gap-2">
          <div className="typing-indicator flex gap-1">
            <span className="w-2 h-2 bg-muted-foreground rounded-full"></span>
            <span className="w-2 h-2 bg-muted-foreground rounded-full"></span>
            <span className="w-2 h-2 bg-muted-foreground rounded-full"></span>
          </div>
          <span className="text-sm text-muted-foreground">ElevenDocs is analyzing...</span>
        </div>
      </div>
    </div>
  );
};