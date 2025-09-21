import { useState, useRef } from "react";
import { Plus, Mic, Wrench, Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SuggestionCards } from "./SuggestionCards";
import { TypingAnimation } from "./TypingAnimation";

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

interface VerionChatProps {
  onProcessingChange?: (isProcessing: boolean) => void;
}

export const VerionChat = ({ onProcessingChange }: VerionChatProps) => {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [hasStartedChat, setHasStartedChat] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!message.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: message.trim(),
      isUser: true,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setMessage("");
    setHasStartedChat(true);
    setIsProcessing(true);
    onProcessingChange?.(true);

    // Simulate AI response
    setTimeout(() => {
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: "I'm Verion, your AI assistant. I understand your query and I'm here to help you with whatever you need. How can I assist you further?",
        isUser: false,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, aiResponse]);
      setIsProcessing(false);
      onProcessingChange?.(false);
    }, 2000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full">
      {!hasStartedChat ? (
        // Initial welcome view
        <div className="flex-1 flex flex-col justify-center items-center px-6 py-8 max-w-4xl mx-auto w-full">
          {/* Main greeting */}
          <div className="text-center mb-12 slide-up">
            <h1 className="text-5xl font-light mb-2">
              Hello, <span className={`font-medium ${isProcessing ? 'text-processing animate-color-shift' : 'text-gradient'}`}>Abhin</span>
            </h1>
          </div>

          {/* Chat input */}
          <div className="w-full max-w-3xl mb-8 slide-up" style={{ animationDelay: "200ms" }}>
            <form onSubmit={handleSubmit}>
              <div className="relative">
                <div className="flex items-center gap-3 bg-chat-input border border-chat-input-border rounded-3xl px-6 py-4 shadow-card-ai hover:border-primary/30 transition-all duration-300 ai-glow">
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-muted-foreground hover:text-primary"
                  >
                    <Plus className="h-4 w-4" />
                  </Button>
                  
                  <Input
                    ref={inputRef}
                    placeholder="Ask Verion"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyPress={handleKeyPress}
                    className="flex-1 border-none bg-transparent text-lg placeholder:text-muted-foreground focus-visible:ring-0 px-0"
                  />
                  
                  <div className="flex items-center gap-2">
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="text-muted-foreground hover:text-primary gap-2 px-3"
                    >
                      <Wrench className="h-4 w-4" />
                      <span className="text-sm">Tools</span>
                    </Button>
                    
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-muted-foreground hover:text-primary"
                    >
                      <Mic className="h-4 w-4" />
                    </Button>
                    
                    {message.trim() && (
                      <Button
                        type="submit"
                        size="icon"
                        className="h-8 w-8 bg-primary hover:bg-primary/90 text-primary-foreground"
                      >
                        <Send className="h-4 w-4" />
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            </form>
          </div>

          {/* Suggestion Cards */}
          <SuggestionCards />
        </div>
      ) : (
        // Chat interface
        <div className="flex-1 flex flex-col h-full">
          {/* Messages area */}
          <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.isUser ? 'justify-end' : 'justify-start'} slide-up`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                    msg.isUser
                      ? 'bg-primary text-primary-foreground ml-12'
                      : 'bg-card border border-border mr-12'
                  }`}
                >
                  {msg.isUser ? (
                    <p className="text-sm leading-relaxed">{msg.text}</p>
                  ) : (
                    <div className="text-sm leading-relaxed text-card-foreground">
                      <TypingAnimation text={msg.text} speed={30} />
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {isProcessing && (
              <div className="flex justify-start slide-up">
                <div className="max-w-[80%] rounded-2xl px-4 py-3 bg-card border border-border mr-12">
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                    <span className="text-sm">Verion is thinking...</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Fixed input at bottom */}
          <div className="border-t border-border bg-background/80 backdrop-blur-sm px-6 py-4">
            <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
              <div className="flex items-center gap-3 bg-chat-input border border-chat-input-border rounded-3xl px-6 py-3 shadow-card-ai hover:border-primary/30 transition-all duration-300">
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 text-muted-foreground hover:text-primary"
                >
                  <Plus className="h-4 w-4" />
                </Button>
                
                <Input
                  ref={inputRef}
                  placeholder="Ask Verion..."
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  className="flex-1 border-none bg-transparent placeholder:text-muted-foreground focus-visible:ring-0 px-0"
                />
                
                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-muted-foreground hover:text-primary"
                  >
                    <Mic className="h-4 w-4" />
                  </Button>
                  
                  {message.trim() && (
                    <Button
                      type="submit"
                      size="icon"
                      className="h-8 w-8 bg-primary hover:bg-primary/90 text-primary-foreground"
                    >
                      <Send className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};