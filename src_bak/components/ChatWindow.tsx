import { useState, useRef, useEffect } from "react";
import { Send, Paperclip, X, FileText, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { AnimatedMessageBubble } from "./AnimatedMessageBubble";
import { ElevenDocsSidebar } from "./ElevenDocsSidebar";
import { useAuth } from "@/contexts/AuthContext";
import { LoginModal } from "./LoginModal";

interface Message {
  id: number;
  text: string;
  isUser: boolean;
  timestamp: string;
  fileInfo?: {
    name: string;
    type: string;
    size: number;
  };
  isTyping?: boolean;
  showTypewriter?: boolean;
}

interface UploadedFile {
  file: File;
  preview: string;
  id: string;
}

interface ChatWindowProps {
  onLogout: () => void;
  initialQuery?: string;
  initialFiles?: File[];
  onBackToHome: () => void;
}

export const ChatWindow = ({ onLogout, initialQuery, initialFiles, onBackToHome }: ChatWindowProps) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "Hello! I'm ElevenDocs, your AI legal assistant. I can help you analyze, summarize, and understand legal documents. How can I assist you today?",
      isUser: false,
      timestamp: "Just now",
      isTyping: false,
      showTypewriter: false
    }
  ]);
  const [inputValue, setInputValue] = useState(initialQuery || "");
  const [isTyping, setIsTyping] = useState(false);
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>(() => {
    if (initialFiles && initialFiles.length > 0) {
      return initialFiles.map(file => ({
        file,
        preview: URL.createObjectURL(file),
        id: Math.random().toString(36).substr(2, 9)
      }));
    }
    return [];
  });
  const [isUploading, setIsUploading] = useState(false);
  const [pendingBotMessage, setPendingBotMessage] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true); // Show sidebar by default on desktop
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { isAuthenticated } = useAuth();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const newFiles: UploadedFile[] = Array.from(files).map(file => ({
      file,
      preview: URL.createObjectURL(file),
      id: Math.random().toString(36).substr(2, 9)
    }));

    setUploadedFiles(prev => [...prev, ...newFiles]);
  };

  const removeFile = (fileId: string) => {
    setUploadedFiles(prev => {
      const fileToRemove = prev.find(f => f.id === fileId);
      if (fileToRemove) {
        URL.revokeObjectURL(fileToRemove.preview);
      }
      return prev.filter(f => f.id !== fileId);
    });
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  // Lock body scroll when sidebar is open
  useEffect(() => {
    if (sidebarOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    
    // Cleanup on unmount
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [sidebarOpen]);

  // Additional scroll when typewriter completes
  const handleTypewriterComplete = () => {
    // Mark the current typing message as completed
    setMessages(prev => prev.map(msg => 
      msg.showTypewriter && !msg.isUser
        ? { ...msg, showTypewriter: false }
        : msg
    ));
    setTimeout(() => scrollToBottom(), 100);
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() && uploadedFiles.length === 0) return;

    // For now, allow sending messages without authentication
    // TODO: Re-enable authentication when backend supports it
    // if (!isAuthenticated) {
    //   setShowLoginModal(true);
    //   return;
    // }

    const newMessage: Message = {
      id: messages.length + 1,
      text: inputValue || (uploadedFiles.length > 0 ? `Uploaded ${uploadedFiles.length} file(s)` : ''),
      isUser: true,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      fileInfo: uploadedFiles.length > 0 ? {
        name: uploadedFiles[0].file.name,
        type: uploadedFiles[0].file.type,
        size: uploadedFiles[0].file.size
      } : undefined,
      isTyping: false,
      showTypewriter: false
    };

    setMessages(prev => [...prev, newMessage]);
    setInputValue("");
    setIsTyping(true);
    setIsUploading(uploadedFiles.length > 0);
    
    // Add single bot message with typing indicator
    const botMessage: Message = {
      id: newMessage.id + 1,
      text: "",
      isUser: false,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      isTyping: true,
      showTypewriter: false
    };
    
    setMessages(prev => [...prev, botMessage]);
    
    try {
      let response;
      
      if (uploadedFiles.length > 0) {
        // Handle file upload
        const formData = new FormData();
        formData.append('input', inputValue || 'Please analyze this document');
        
        // Process each uploaded file
        for (const uploadedFile of uploadedFiles) {
          const fileType = uploadedFile.file.type.split('/')[1] || 
                          uploadedFile.file.name.split('.').pop()?.toLowerCase() || 'txt';
          formData.append('file_data', uploadedFile.file);
          formData.append('file_type', fileType);
        }

        response = await fetch('http://127.0.0.1:8002/invoke', {
          method: 'POST',
          body: formData
        });
      } else {
        // Handle text-only message
        response = await fetch('http://127.0.0.1:8002/invoke', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ text: newMessage.text })
        });
      }

      if (!response.ok) {
        throw new Error('Request failed');
      }

      const data = await response.json();

      // Use the Python agent's response directly
      const botText = data.response || data.output || 'No response from agent';

      // Update the existing bot message to show typewriter effect
      setMessages(prev => prev.map(msg => 
        msg.isTyping && !msg.isUser 
          ? {
              ...msg,
              text: botText,
              isTyping: false,
              showTypewriter: true
            }
          : msg
      ));
    } catch (err) {
      // Update the existing bot message to show error
      setMessages(prev => prev.map(msg => 
        msg.isTyping && !msg.isUser 
          ? {
              ...msg,
              text: '⚠️ Unable to process your request. Please try again.',
              isTyping: false,
              showTypewriter: false
            }
          : msg
      ));
    } finally {
      setIsTyping(false);
      setIsUploading(false);
      setUploadedFiles([]); // Clear uploaded files after sending
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleNewChat = () => {
    setMessages([{
      id: 1,
      text: "Hello! I'm ElevenDocs, your AI legal assistant. I can help you analyze, summarize, and understand legal documents. How can I assist you today?",
      isUser: false,
      timestamp: "Just now",
      isTyping: false,
      showTypewriter: false
    }]);
  };

  // Process initial query and files if provided
  useEffect(() => {
    if ((initialQuery && initialQuery.trim()) || (initialFiles && initialFiles.length > 0)) {
      setTimeout(() => {
        handleSendMessage();
      }, 500);
    }
  }, [initialQuery, initialFiles]);

  return (
    <div className="flex h-screen bg-background relative">
      {/* Fixed Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 w-64 sidebar-container sidebar-transition ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        <ElevenDocsSidebar 
          onLogout={onLogout} 
          onNewChat={handleNewChat} 
          onBackToHome={onBackToHome}
          onCloseSidebar={() => setSidebarOpen(false)}
        />
      </div>
      
      {/* Overlay when sidebar is open */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/20 backdrop-blur-sm z-40 overlay-transition"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      
      {/* Main Content */}
      <div className={`flex-1 flex flex-col main-content-transition ${
        sidebarOpen ? 'ml-64' : 'ml-0'
      }`}>
        {/* Header with Hamburger Menu - Always visible */}
        <div className="border-b border-border bg-background p-4">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="h-8 w-8 p-0"
            >
              <Menu size={16} />
            </Button>
            <div className="flex items-center gap-2">
              <div className="h-6 w-6 bg-primary rounded flex items-center justify-center">
                <span className="text-primary-foreground font-bold text-xs">11D</span>
              </div>
              <h1 className="text-lg font-semibold">ElevenDocs</h1>
            </div>
          </div>
        </div>

        {/* Main Chat Area */}
        <div className="flex-1 overflow-y-auto bg-chat-bg">
          <div className="max-w-3xl mx-auto px-4">
            {messages.map((message, index) => {
              // Check if this is the latest bot message that should show typing animation
              const isLatestBotMessage = !message.isUser && 
                messages.slice(index + 1).every(msg => msg.isUser || !msg.showTypewriter) &&
                message.showTypewriter;
              
              return (
                <AnimatedMessageBubble
                  key={message.id}
                  message={message.text}
                  isUser={message.isUser}
                  timestamp={message.timestamp}
                  fileInfo={message.fileInfo}
                  isTyping={message.isTyping}
                  showTypewriter={message.showTypewriter}
                  onTypewriterComplete={handleTypewriterComplete}
                  isLatestBotMessage={isLatestBotMessage}
                />
              );
            })}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="border-t border-border bg-background p-4">
          <div className="max-w-3xl mx-auto px-4">
            {/* File Preview Area */}
            {uploadedFiles.length > 0 && (
              <div className="mb-3 p-3 bg-muted rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <FileText size={16} className="text-muted-foreground" />
                  <span className="text-sm font-medium">Attached Files:</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {uploadedFiles.map((file) => (
                    <div
                      key={file.id}
                      className="flex items-center gap-2 bg-background px-3 py-2 rounded-md border text-sm"
                    >
                      <FileText size={14} className="text-muted-foreground" />
                      <span className="truncate max-w-[200px]">{file.file.name}</span>
                      <span className="text-muted-foreground text-xs">
                        ({(file.file.size / 1024).toFixed(1)} KB)
                      </span>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-6 w-6 p-0 hover:bg-destructive hover:text-destructive-foreground"
                        onClick={() => removeFile(file.id)}
                      >
                        <X size={12} />
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="relative flex items-end gap-3">
              <div className="flex-1 relative">
                <Textarea
                  ref={textareaRef}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask ElevenDocs about your legal document..."
                  className="min-h-[44px] max-h-32 resize-none bg-input-bg border-input-border focus:border-input-border-focus input-glow pr-12 transition-all duration-300"
                  rows={1}
                />
                <Button
                  size="sm"
                  variant="ghost"
                  className="absolute right-2 top-2 h-8 w-8 p-0 hover:bg-muted"
                  onClick={triggerFileInput}
                  title="Attach file"
                >
                  <Paperclip size={16} className="text-muted-foreground" />
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".pdf,.txt,.doc,.docx,.png,.jpg,.jpeg"
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </div>
              
              <Button
                onClick={handleSendMessage}
                disabled={(!inputValue.trim() && uploadedFiles.length === 0) || isTyping || isUploading}
                className="h-11 w-11 p-0 bg-primary hover:bg-primary-hover text-primary-foreground transition-colors disabled:opacity-50"
              >
                {isUploading ? (
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-current border-t-transparent" />
                ) : (
                  <Send size={16} />
                )}
              </Button>
            </div>
            
            <p className="text-xs text-muted-foreground text-center mt-2">
              ElevenDocs can analyze legal documents. Always consult with a qualified attorney for legal advice.
            </p>
          </div>
        </div>
      </div>

      {/* Login Modal */}
      {showLoginModal && (
        <LoginModal 
          isOpen={showLoginModal}
          onClose={() => setShowLoginModal(false)}
          onSuccess={() => setShowLoginModal(false)}
        />
      )}
    </div>
  );
};