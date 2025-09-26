import { useState, useRef, useEffect } from "react";
import { Paperclip, Send, X, FileText, Menu, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { UserMenu } from "./UserMenu";
import { AnimatedMessageBubble } from "./AnimatedMessageBubble";
import { ElevenDocsSidebar as VerionSidebar } from "./ElevenDocsSidebar";
import { ChatGPTInput } from "./ChatGPTInput";
import { useAuth } from "@/contexts/AuthContext";

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

interface LandingPageProps {
  onStartChat: (query: string, files?: File[]) => void;
  isAuthenticated: boolean;
  onLogout: () => void;
}

export const LandingPage = ({ onStartChat, isAuthenticated, onLogout }: LandingPageProps) => {
  const { logout } = useAuth();
  const [query, setQuery] = useState("");
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
  text: "Hello! I'm Verion, your AI legal assistant. I can help you analyze, summarize, and understand legal documents. How can I assist you today?",
      isUser: false,
      timestamp: "Just now",
      isTyping: false,
      showTypewriter: false
    }
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true); // Sidebar visible by default
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  const triggerFileInput = () => {
    fileInputRef.current?.click();
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() && uploadedFiles.length === 0) return;

    // Show chat interface
    setShowChat(true);

    const newMessage: Message = {
      id: messages.length + 1,
      text: query || (uploadedFiles.length > 0 ? `Uploaded ${uploadedFiles.length} file(s)` : ''),
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
    setQuery("");
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
        formData.append('input', query || 'Please analyze this document');
        
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
        // Handle text-only message (send to /invoke)
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
      handleSubmit(e);
    }
  };

  const handleNewChat = () => {
    setMessages([{
      id: 1,
  text: "Hello! I'm Verion, your AI legal assistant. I can help you analyze, summarize, and understand legal documents. How can I assist you today?",
      isUser: false,
      timestamp: "Just now",
      isTyping: false,
      showTypewriter: false
    }]);
    setShowChat(false);
    setQuery("");
  };

  const handleTypewriterComplete = () => {
    setMessages(prev => prev.map(msg => 
      msg.showTypewriter && !msg.isUser
        ? { ...msg, showTypewriter: false }
        : msg
    ));
    setTimeout(() => scrollToBottom(), 100);
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

  return (
    <div className="flex h-screen bg-background relative">
      {/* Fixed Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 w-64 sidebar-container sidebar-transition ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
  <VerionSidebar 
          onLogout={logout} 
          onNewChat={handleNewChat}
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
      <div className={`flex-1 flex flex-col main-content-transition ${sidebarOpen ? 'ml-64' : 'ml-0'}`}> 
        {/* Menu Bar Only (no banner, no extra header) */}
        <div className="p-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="h-8 w-8 p-0 group"
          >
            <Menu 
              size={16} 
              className="transition-transform duration-300 group-hover:rotate-90 group-hover:scale-125" 
            />
          </Button>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          {!showChat ? (
            // Welcome and search bar centered
            <main className="flex-1 flex flex-col items-center justify-center px-6 animate-fade-in">
              <div className="w-full max-w-2xl text-center space-y-8 flex flex-col items-center justify-center">
                <div className="space-y-4">
                  <h2 className="text-4xl md:text-5xl font-bold text-foreground">
                    Welcome to Verion
                  </h2>
                  <p className="text-lg text-muted-foreground font-semibold">
                    AI-powered legal document analysis at your fingertips
                  </p>
                </div>
                {/* File Preview Area */}
                {uploadedFiles.length > 0 && (
                  <div className="mb-6 p-4 bg-muted rounded-lg max-w-lg mx-auto">
                    <div className="flex items-center gap-2 mb-3">
                      <span className="text-sm font-medium">Selected Files:</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {uploadedFiles.map((file) => (
                        <div
                          key={file.id}
                          className="flex items-center gap-2 bg-background px-3 py-2 rounded-md border text-sm"
                        >
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
                            ×
                          </Button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {/* Clean Search Bar */}
                <div className="flex flex-col items-center mt-8">
                  <form onSubmit={handleSubmit} className="w-full max-w-6xl">
                    <div className="relative">
                      <div className="flex items-center rounded-full border border-gray-300 bg-white shadow-lg" style={{ width: '100%' }}>
                        <input
                          value={query}
                          onChange={(e) => setQuery(e.target.value)}
                          placeholder="Ask Verion or upload files..."
                          className="flex-1 h-14 px-6 text-lg bg-transparent border-0 outline-none placeholder:text-gray-500 text-gray-900"
                          style={{ minWidth: '600px', background: 'transparent', border: 'none', outline: 'none', boxShadow: 'none' }}
                        />
                        <div className="flex items-center pr-4 gap-2">
                          <Button
                            type="submit"
                            size="sm"
                            variant="ghost"
                            className="h-8 w-8 p-0 rounded-full text-blue-500 hover:text-blue-700"
                            title="Send"
                            disabled={!query.trim()}
                          >
                            <Send size={20} />
                          </Button>
                          <Button
                            type="button"
                            size="sm"
                            variant="ghost"
                            onClick={triggerFileInput}
                            className="h-8 w-8 p-0 rounded-full text-gray-500 hover:text-gray-700"
                            title="Attach file"
                          >
                            <Paperclip size={20} />
                          </Button>
                        </div>
                      </div>
                    </div>
                    <input
                      ref={fileInputRef}
                      type="file"
                      multiple
                      accept=".pdf,.txt,.doc,.docx,.png,.jpg,.jpeg"
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                  </form>
                </div>
                
              </div>
            </main>
          ) : (
            // Chat interface with ChatGPT-style sticky input
            <>
              <div className="flex-1 overflow-y-auto bg-chat-bg chat-scroll-area">
                <div className="max-w-3xl mx-auto px-4">
                  {messages.map((message, index) => {
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
              
              {/* ChatGPT-style Sticky Input */}
              <ChatGPTInput
                query={query}
                setQuery={setQuery}
                onSubmit={handleSubmit}
                onKeyPress={handleKeyPress}
                uploadedFiles={uploadedFiles}
                onFileSelect={handleFileSelect}
                onRemoveFile={removeFile}
                triggerFileInput={triggerFileInput}
                isTyping={isTyping}
                isUploading={isUploading}
                fileInputRef={fileInputRef}
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
};