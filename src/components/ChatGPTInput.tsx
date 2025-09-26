import { useState, useRef, useEffect } from "react";
import { Paperclip, Send, X, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";

interface UploadedFile {
  file: File;
  preview: string;
  id: string;
}

interface ChatGPTInputProps {
  query: string;
  setQuery: (query: string) => void;
  onSubmit: (e: React.FormEvent) => void;
  onKeyPress: (e: React.KeyboardEvent) => void;
  uploadedFiles: UploadedFile[];
  onFileSelect: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onRemoveFile: (fileId: string) => void;
  triggerFileInput: () => void;
  isTyping: boolean;
  isUploading: boolean;
  fileInputRef: React.RefObject<HTMLInputElement>;
}

export const ChatGPTInput = ({
  query,
  setQuery,
  onSubmit,
  onKeyPress,
  uploadedFiles,
  onFileSelect,
  onRemoveFile,
  triggerFileInput,
  isTyping,
  isUploading,
  fileInputRef
}: ChatGPTInputProps) => {
  const [isFocused, setIsFocused] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
  }, [query]);

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-background border-t border-border z-50">
      <div className="max-w-4xl mx-auto p-4">
        {/* File Preview Area */}
        {uploadedFiles.length > 0 && (
          <div className="mb-4 p-4 bg-muted/50 rounded-xl border border-border/50">
            <div className="flex items-center gap-2 mb-3">
              <FileText size={16} className="text-muted-foreground" />
              <span className="text-sm font-medium text-foreground">Attached Files:</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {uploadedFiles.map((file) => (
                <div
                  key={file.id}
                  className="flex items-center gap-2 bg-background px-3 py-2 rounded-lg border text-sm"
                >
                  <FileText size={14} className="text-muted-foreground" />
                  <span className="truncate max-w-[200px] text-foreground">{file.file.name}</span>
                  <span className="text-muted-foreground text-xs">
                    ({(file.file.size / 1024).toFixed(1)} KB)
                  </span>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 w-6 p-0 hover:bg-destructive hover:text-destructive-foreground"
                    onClick={() => onRemoveFile(file.id)}
                  >
                    <X size={12} />
                  </Button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ChatGPT-style Input */}
        <form onSubmit={onSubmit} className="relative">
          <div className={`chatgpt-input-container ${isFocused ? 'focused' : ''}`}>
            <textarea
              ref={textareaRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={onKeyPress}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              placeholder="Ask Verion about your legal document..."
              className="chatgpt-textarea"
              rows={1}
              disabled={isTyping || isUploading}
            />
            
            {/* File Upload Button */}
            <button
              type="button"
              onClick={triggerFileInput}
              className="chatgpt-file-btn"
              disabled={isTyping || isUploading}
              title="Attach file"
            >
              <Paperclip size={18} />
            </button>

            {/* Send Button */}
            <button
              type="submit"
              className="chatgpt-send-btn"
              disabled={(!query.trim() && uploadedFiles.length === 0) || isTyping || isUploading}
            >
              {isUploading ? (
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-current border-t-transparent" />
              ) : (
                <Send size={18} />
              )}
            </button>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.txt,.doc,.docx,.png,.jpg,.jpeg"
            onChange={onFileSelect}
            className="hidden"
          />
        </form>

        {/* Disclaimer */}
        <p className="text-xs text-muted-foreground text-center mt-3">
          Verion can analyze legal documents. Always consult with a qualified attorney for legal advice.
        </p>
      </div>
    </div>
  );
};
